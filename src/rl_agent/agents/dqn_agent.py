import random
from collections import deque, namedtuple
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class QNetwork(nn.Module):
    def __init__(self, input_dim: int, n_actions: int):
        super().__init__()
        hidden = 512
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
    def forward(self, x): return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity: int): self.buffer = deque(maxlen=capacity)
    def __len__(self): return len(self.buffer)
    def push(self, *args): self.buffer.append(Transition(*args))
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

class DQNAgent:
    def __init__(
        self, state_dim: int, n_actions: int,
        learning_rate: float=1e-3, gamma: float=0.99,
        epsilon_start: float=1.0, epsilon_end: float=0.05, epsilon_decay_steps: int=20000,
        memory_size: int=100000, batch_size: int=128,
        target_update_every: int=1000, gradient_clip_norm: float=1.0, seed: int=42
    ):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict()); self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.memory = ReplayBuffer(memory_size)
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.gradient_clip_norm = gradient_clip_norm

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = max(1, epsilon_decay_steps)
        self.total_steps = 0
        self.n_actions = n_actions

    def epsilon(self):
        frac = min(1.0, self.total_steps / self.epsilon_decay_steps)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * frac

    def select_action(self, state: np.ndarray) -> int:
        self.total_steps += 1
        if random.random() < self.epsilon():
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy_net(s)
            return int(q.argmax(dim=1).item())

    def remember(self, transition: Transition): self.memory.push(*transition)

    def train_step(self) -> Optional[float]:
        if len(self.memory) < self.batch_size: return None
        batch = self.memory.sample(self.batch_size)

        state = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        action = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        done = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.policy_net(state).gather(1, action)
        with torch.no_grad():
            q_next = self.target_net(next_state).max(dim=1, keepdim=True)[0]
            target = reward + (1.0 - done) * self.gamma * q_next

        loss = F.smooth_l1_loss(q_sa, target)
        self.optimizer.zero_grad(); loss.backward()
        if self.gradient_clip_norm and self.gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip_norm)
        self.optimizer.step()

        if self.total_steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.item())

    def save(self, path: str):
        torch.save({"model_state_dict": self.policy_net.state_dict(),
                    "total_steps": self.total_steps,
                    "n_actions": self.n_actions}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["model_state_dict"])
        self.target_net.load_state_dict(ckpt["model_state_dict"])
        self.total_steps = ckpt.get("total_steps", 0)
        self.n_actions = ckpt.get("n_actions", self.n_actions)
