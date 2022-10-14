from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

from tianshou.data import Batch
from tianshou.data.batch import _alloc_by_keys_diff, _create_value
from tianshou.data.utils.converter import from_hdf5, to_hdf5


class ReplayBuffer:
    """:class:`~tianshou.data.ReplayBuffer` stores data generated from interaction \
    between the policy and environment.

    ReplayBuffer can be considered as a specialized form (or management) of Batch. It
    stores all the data in a batch with circular-queue style.

    For the example usage of ReplayBuffer, please check out Section Buffer in
    :doc:`/tutorials/concepts`.

    :param int size: the maximum size of replay buffer.
    :param int stack_num: the frame-stack sampling argument, should be greater than or
        equal to 1. Default to 1 (no stacking).
    :param bool ignore_obs_next: whether to store obs_next. Default to False.
    :param bool save_only_last_obs: only save the last obs/obs_next when it has a shape
        of (timestep, ...) because of temporal stacking. Default to False.
    :param bool sample_avail: the parameter indicating sampling only available index
        when using frame-stack sampling method. Default to False.
    """

    _reserved_keys = ("obs", "act", "rew", "done", "obs_next", "info", "policy")

    def __init__(
        self,
        size: int,
        stack_num: int = 1,
        ignore_obs_next: bool = False,
        save_only_last_obs: bool = False,
        sample_avail: bool = False,
        **kwargs: Any,  # otherwise PrioritizedVectorReplayBuffer will cause TypeError
    ) -> None:
        self.options: Dict[str, Any] = {
            "stack_num": stack_num,
            "ignore_obs_next": ignore_obs_next,
            "save_only_last_obs": save_only_last_obs,
            "sample_avail": sample_avail,
        }
        super().__init__()
        self.maxsize = int(size)
        assert stack_num > 0, "stack_num should be greater than 0"
        self.stack_num = stack_num
        self._indices = np.arange(size)
        self._save_obs_next = not ignore_obs_next
        self._save_only_last_obs = save_only_last_obs
        self._sample_avail = sample_avail
        self._meta: Batch = Batch()
        self._ep_rew: Union[float, np.ndarray]
        self.reset()

    def __len__(self) -> int:
        """Return len(self)."""
        return self._size

    def __repr__(self) -> str:
        """Return str(self)."""
        return self.__class__.__name__ + self._meta.__repr__()[5:]

    def __getattr__(self, key: str) -> Any:
        """Return self.key."""
        try:
            return self._meta[key]
        except KeyError as exception:
            raise AttributeError from exception

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Unpickling interface.

        We need it because pickling buffer does not work out-of-the-box
        ("buffer.__getattr__" is customized).
        """
        self.__dict__.update(state)

    def __setattr__(self, key: str, value: Any) -> None:
        """Set self.key = value."""
        assert (key not in self._reserved_keys
                ), "key '{}' is reserved and cannot be assigned".format(key)
        super().__setattr__(key, value)

    def save_hdf5(self, path: str, compression: Optional[str] = None) -> None:
        """Save replay buffer to HDF5 file."""
        with h5py.File(path, "w") as f:
            to_hdf5(self.__dict__, f, compression=compression)

    @classmethod
    def load_hdf5(cls, path: str, device: Optional[str] = None) -> "ReplayBuffer":
        """Load replay buffer from HDF5 file."""
        with h5py.File(path, "r") as f:
            buf = cls.__new__(cls)
            buf.__setstate__(from_hdf5(f, device=device))  # type: ignore
        return buf

    @classmethod
    def from_data(
        cls, obs: h5py.Dataset, act: h5py.Dataset, rew: h5py.Dataset,
        done: h5py.Dataset, obs_next: h5py.Dataset
    ) -> "ReplayBuffer":
        size = len(obs)
        assert all(len(dset) == size for dset in [obs, act, rew, done, obs_next]), \
            "Lengths of all hdf5 datasets need to be equal."
        buf = cls(size)
        if size == 0:
            return buf
        batch = Batch(obs=obs, act=act, rew=rew, done=done, obs_next=obs_next)
        buf.set_batch(batch)
        buf._size = size
        return buf

    def reset(self, keep_statistics: bool = False) -> None:
        """Clear all the data in replay buffer and episode statistics."""
        self.last_index = np.array([0])
        self._index = self._size = 0
        if not keep_statistics:
            self._ep_rew, self._ep_len, self._ep_idx = 0.0, 0, 0

    def set_batch(self, batch: Batch) -> None:
        """Manually choose the batch you want the ReplayBuffer to manage."""
        assert len(batch) == self.maxsize and set(batch.keys()).issubset(
            self._reserved_keys
        ), "Input batch doesn't meet ReplayBuffer's data form requirement."
        self._meta = batch

    def unfinished_index(self) -> np.ndarray:
        """Return the index of unfinished episode."""
        last = (self._index - 1) % self._size if self._size else 0
        return np.array([last] if not self.done[last] and self._size else [], int)

    def prev(self, index: Union[int, np.ndarray]) -> np.ndarray:
        """Return the index of previous transition.

        The index won't be modified if it is the beginning of an episode.
        """
        index = (index - 1) % self._size
        end_flag = self.done[index] | (index == self.last_index[0])
        return (index + end_flag) % self._size

    def next(self, index: Union[int, np.ndarray]) -> np.ndarray:
        """Return the index of next transition.

        The index won't be modified if it is the end of an episode.
        """
        end_flag = self.done[index] | (index == self.last_index[0])
        return (index + (1 - end_flag)) % self._size

    def update(self, buffer: "ReplayBuffer") -> np.ndarray:
        """Move the data from the given buffer to current buffer.

        Return the updated indices. If update fails, return an empty array.
        """
        if len(buffer) == 0 or self.maxsize == 0:
            return np.array([], int)
        stack_num, buffer.stack_num = buffer.stack_num, 1
        from_indices = buffer.sample_indices(0)  # get all available indices
        buffer.stack_num = stack_num
        if len(from_indices) == 0:
            return np.array([], int)
        to_indices = []
        for _ in range(len(from_indices)):
            to_indices.append(self._index)
            self.last_index[0] = self._index
            self._index = (self._index + 1) % self.maxsize
            self._size = min(self._size + 1, self.maxsize)
        to_indices = np.array(to_indices)
        if self._meta.is_empty():
            self._meta = _create_value(  # type: ignore
                buffer._meta, self.maxsize, stack=False)
        self._meta[to_indices] = buffer._meta[from_indices]
        return to_indices

    def _add_index(self, rew: Union[float, np.ndarray],
                   done: bool) -> Tuple[int, Union[float, np.ndarray], int, int]:
        """Maintain the buffer's state after adding one data batch.

        Return (index_to_be_modified, episode_reward, episode_length,
        episode_start_index).
        """
        self.last_index[0] = ptr = self._index
        self._size = min(self._size + 1, self.maxsize)
        self._index = (self._index + 1) % self.maxsize

        self._ep_rew += rew
        self._ep_len += 1

        if done:
            result = ptr, self._ep_rew, self._ep_len, self._ep_idx
            self._ep_rew, self._ep_len, self._ep_idx = 0.0, 0, self._index
            return result
        else:
            return ptr, self._ep_rew * 0.0, 0, self._ep_idx

    def add(
        self,
        batch: Batch,
        buffer_ids: Optional[Union[np.ndarray, List[int]]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Add a batch of data into replay buffer.

        :param Batch batch: the input data batch. Its keys must belong to the 7
            reserved keys, and "obs", "act", "rew", "done" is required.
        :param buffer_ids: to make consistent with other buffer's add function; if it
            is not None, we assume the input batch's first dimension is always 1.

        Return (current_index, episode_reward, episode_length, episode_start_index). If
        the episode is not finished, the return value of episode_length and
        episode_reward is 0.
        """
        # preprocess batch
        new_batch = Batch()
        for key in set(self._reserved_keys).intersection(batch.keys()):
            new_batch.__dict__[key] = batch[key]
        batch = new_batch
        assert set(["obs", "act", "rew", "done"]).issubset(batch.keys())
        stacked_batch = buffer_ids is not None
        if stacked_batch:
            assert len(batch) == 1
        if self._save_only_last_obs:
            batch.obs = batch.obs[:, -1] if stacked_batch else batch.obs[-1]
        if not self._save_obs_next:
            batch.pop("obs_next", None)
        elif self._save_only_last_obs:
            batch.obs_next = (
                batch.obs_next[:, -1] if stacked_batch else batch.obs_next[-1]
            )
        # get ptr
        if stacked_batch:
            rew, done = batch.rew[0], batch.done[0]
        else:
            rew, done = batch.rew, batch.done
        ptr, ep_rew, ep_len, ep_idx = list(
            map(lambda x: np.array([x]), self._add_index(rew, done))
        )
        try:
            self._meta[ptr] = batch
        except ValueError:
            stack = not stacked_batch
            batch.rew = batch.rew.astype(float)
            batch.done = batch.done.astype(bool)
            if self._meta.is_empty():
                self._meta = _create_value(  # type: ignore
                    batch, self.maxsize, stack)
            else:  # dynamic key pops up in batch
                _alloc_by_keys_diff(self._meta, batch, self.maxsize, stack)
            self._meta[ptr] = batch
        return ptr, ep_rew, ep_len, ep_idx

    def sample_indices(self, batch_size: int) -> np.ndarray:
        """Get a random sample of index with size = batch_size.

        Return all available indices in the buffer if batch_size is 0; return an empty
        numpy array if batch_size < 0 or no available index can be sampled.
        """
        if self.stack_num == 1 or not self._sample_avail:  # most often case
            if batch_size > 0:
                return np.random.choice(self._size, batch_size)
            elif batch_size == 0:  # construct current available indices
                return np.concatenate(
                    [np.arange(self._index, self._size),
                     np.arange(self._index)]
                )
            else:
                return np.array([], int)
        else:
            if batch_size < 0:
                return np.array([], int)
            all_indices = prev_indices = np.concatenate(
                [np.arange(self._index, self._size),
                 np.arange(self._index)]
            )
            for _ in range(self.stack_num - 2):
                prev_indices = self.prev(prev_indices)
            all_indices = all_indices[prev_indices != self.prev(prev_indices)]
            if batch_size > 0:
                return np.random.choice(all_indices, batch_size)
            else:
                return all_indices

    def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray]:
        """Get a random sample from buffer with size = batch_size.

        Return all the data in the buffer if batch_size is 0.

        :return: Sample data and its corresponding index inside the buffer.
        """
        indices = self.sample_indices(batch_size)
        return self[indices], indices

    def sample_low_reward_indices(self, batch_size:int):
        observations = self.__getattr__('obs')
        rewards = self.__getattr__('rew')
        # print('first two invalid rewards:{}, {}'.format(rewards[self.buffer._size], rewards[self.buffer._size+1]))
        rewards = rewards[:self._size]
        # print('last two valid rewards:{}, {}'.format(rewards[-2], rewards[-1]))
        # abs_rewards = np.absolute(rewards)
        if self._size < batch_size:
            index = np.argpartition(rewards, self._size)[:self._size]
        else:
            amount = max(int(self._size *0.005), batch_size)
            index = np.argpartition(rewards, amount)[:amount]
        selected_index = np.random.choice(index,batch_size)
        # print(abs_rewards[index])
        return observations, rewards, selected_index



    def get(
        self,
        index: Union[int, List[int], np.ndarray],
        key: str,
        default_value: Any = None,
        stack_num: Optional[int] = None,
    ) -> Union[Batch, np.ndarray]:
        """Return the stacked result.

        E.g., if you set ``key = "obs", stack_num = 4, index = t``, it returns the
        stacked result as ``[obs[t-3], obs[t-2], obs[t-1], obs[t]]``.

        :param index: the index for getting stacked data.
        :param str key: the key to get, should be one of the reserved_keys.
        :param default_value: if the given key's data is not found and default_value is
            set, return this default_value.
        :param int stack_num: Default to self.stack_num.
        """
        if key not in self._meta and default_value is not None:
            return default_value
        val = self._meta[key]
        if stack_num is None:
            stack_num = self.stack_num
        try:
            if stack_num == 1:  # the most often case
                return val[index]
            stack: List[Any] = []
            if isinstance(index, list):
                indices = np.array(index)
            else:
                indices = index  # type: ignore
            for _ in range(stack_num):
                stack = [val[indices]] + stack
                indices = self.prev(indices)
            if isinstance(val, Batch):
                return Batch.stack(stack, axis=indices.ndim)
            else:
                return np.stack(stack, axis=indices.ndim)
        except IndexError as exception:
            if not (isinstance(val, Batch) and val.is_empty()):
                raise exception  # val != Batch()
            return Batch()

    def __getitem__(self, index: Union[slice, int, List[int], np.ndarray]) -> Batch:
        """Return a data batch: self[index].

        If stack_num is larger than 1, return the stacked obs and obs_next with shape
        (batch, len, ...).
        """
        if isinstance(index, slice):  # change slice to np array
            # buffer[:] will get all available data
            indices = self.sample_indices(0) if index == slice(None) \
                else self._indices[:len(self)][index]
        else:
            indices = index  # type: ignore
        # raise KeyError first instead of AttributeError,
        # to support np.array([ReplayBuffer()])
        obs = self.get(indices, "obs")
        if self._save_obs_next:
            obs_next = self.get(indices, "obs_next", Batch())
        else:
            obs_next = self.get(self.next(indices), "obs", Batch())
        return Batch(
            obs=obs,
            act=self.act[indices],
            rew=self.rew[indices],
            done=self.done[indices],
            obs_next=obs_next,
            info=self.get(indices, "info", Batch()),
            policy=self.get(indices, "policy", Batch()),
        )

    def replace_transaction(self, index, obs, act, rew, done, obs_next, info, policy):
        self.obs[index] = obs
        self.act[index] = act
        self.rew[index] = rew
        self.done[index] = done
        self.obs_next[index] = obs_next
        # self.info[index] = info
        self.policy[index] = policy

    # def __setitem__(self, index: Union[int, List[int], np.ndarray], reserved_key) -> None:
    #     """
    #     Set a data batch in the certain position of Batch
    #     :param index: position
    #     :param reserved_key: {"obs", "act", "rew", "done", "obs_next", "info", "policy"}
    #     :return:
    #     """
    #     states = buffer.__getattr__('obs')
    #     actions = buffer.__getattr__('act')
    #     rewards = buffer.__getattr__('rew')
    #     done = buffer.__getattr__('done')
    #     obs_next = buffer.__getattr__('obs_next')
    #     info = buffer.__getattr__('info')
    #     policy = buffer.__getattr__('policy')
    #
    #     states[index] = reserved_key['obs']
    #     actions[index] = reserved_key['act']
    #     rewards[index] = reserved_key['rew']
    #     done[index] = reserved_key['done']
    #     obs_next[index] = reserved_key['obs_next']
    #     info[index] = reserved_key['info']
    #     policy[index] = reserved_key['policy']
    #
    #     batch = Batch(obs = states, act= actions, rew = rewards, done = done, obs_next = obs_next, info = info, policy = policy)
    #     self.set_batch(batch)
    #
    #
    #
    # def take_place(self,  buffer):
    #     from_indices = buffer.sample_indices(0)  # get all available indices
    #     print(from_indices)

# buffer = ReplayBuffer(size=10)
#
# # b={'c': [2., 'st'], 'd': [1., 0.]}
# # data = Batch(a=[False,  True], b=b)
# # print(data)
# data = Batch(obs=[1,2,1],act=2,rew=3,done=True, obs_next=[1,1,1], info='None', policy=[[1,2],[4,5]])
# data2 = Batch(obs=[4,5,2],act=6,rew=-3,done=False, obs_next=[2,2,2], info='None', policy=[[1,2],[4,5]])
# data3 = Batch(obs=[4,5,3],act=6,rew=1,done=False, obs_next=[3,3,3], info='None', policy=[[1,2],[4,5]])
# data4 = Batch(obs=[4,5,4],act=6,rew=7,done=False, obs_next=[4,4,4], info='None', policy=[[1,2],[4,5]])
# data5 = Batch(obs=[4,5,5],act=6,rew=-8,done=False, obs_next=[5,5,5], info='None', policy=[[1,2],[4,5]])
# data6 = Batch(obs=[4,5,6],act=6,rew=-12,done=False, obs_next=[6,6,6], info='None', policy=[[1,2],[4,5]])
# # print(data)
# buffer.add(data)
# buffer.add(data2)
# buffer.add(data3)
# buffer.add(data4)
# buffer.add(data5)
# buffer.add(data6)

# buffer.__setitem__(2,obs=[0,0,0],act=0,rew=0,done=False, obs_next=[0,0,0], info='None', policy=[[0,0],[0,0]])
# # a = buffer.__getitem__(2)
# print(buffer)
# data6.obs = [2,2,2]
# print(data6)

# new_buffer = ReplayBuffer(size=100)
# test = Batch(obs=[0,0,0],act=0,rew=0,done=False, obs_next=[0,0,0], info='None', policy=[[0,0],[0,0]])


# a = buffer.__getitem__(3)
# print(a)
#
#
# buffer.set_batch(test)
# print(buffer)
# reserved_key = {}
# reserved_key['obs'] = [9,9,9]
# reserved_key['act'] = 9
# reserved_key['rew'] = -99
# reserved_key['done'] = True
# reserved_key['obs_next'] = [10,10,10]
# reserved_key['info'] = 'Success'
# reserved_key['policy'] = [[-1,-2],[-3,-4]]
#
# buffer.__setitem__([3,4],reserved_key)
#
# b = buffer.__getitem__(4)
# print(b)



# print(buffer)
# # buffer.take_place(buffer)
# buffer_2 = buffer.update(buffer)
# print([buffer_2])

# a,b = buffer.sample(1)
# print(a)
# print(b)

# reward = buffer.__getattr__('rew')
# print(reward[2])
# buffer.__setattr__('rew',[8,8,8,8])
# reward = buffer.__getattr__('rew')
# print(reward[2])
# k = 3
# b = np.argpartition(reward, k)[:k]
# print(b)
# r = buffer.__getitem__(b)
# print(r.obs)
# print(r.rew)