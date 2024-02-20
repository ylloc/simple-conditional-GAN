import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder, MNIST
from torchvision import transforms
from torch import autograd
import torch.utils.data
from tqdm.auto import tqdm

BATCH_SIZE = 256

def get_dataset() -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    data_loader = torch.utils.data.DataLoader(
        MNIST('data', train=True, download=True, transform=transform),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    return data_loader

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), 784)
        x = torch.cat([x, self.emb(labels)], 1)
        return self.model(x).squeeze()

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), 100)
        y = torch.cat([x, self.emb(labels)], 1)
        return self.model(y).view(y.size(0), 28, 28)

EPOCHS = 50
TIMES = 5

if __name__ == "__main__":
    G, D = Generator().cuda(), Discriminator().cuda()
    loss_fn = nn.BCELoss()
    d_optim = torch.optim.Adam(D.parameters(), lr=1e-4)
    g_optim = torch.optim.Adam(G.parameters(), lr=1e-4)
    data_loader = get_dataset()
    pbar = tqdm(range(EPOCHS))
    G.train()
    D.train()

    for epoch in pbar:
        pbar.update(1)
        for i, (images, labels) in enumerate(data_loader):
            real_images, labels = torch.Tensor(images).cuda(), torch.Tensor(labels).cuda()

            for _ in range(TIMES):
                d_optim.zero_grad()
                """
                loss_D = E_{x ~ real} [log(D(x))] + E_{x ~ fake} [log(1 - D(x))
                """
                real_out = D(real_images, labels)
                loss_real = loss_fn(real_out, torch.Tensor(torch.ones(BATCH_SIZE)).cuda())

                noice = torch.Tensor(torch.randn(BATCH_SIZE, 100)).cuda()
                fake_labels = torch.Tensor(torch.LongTensor(np.random.randint(0, 10, BATCH_SIZE))).cuda()
                fake_images = G(noice, fake_labels)
                fake_validity = D(fake_images, fake_labels)
                loss_fake = loss_fn(fake_validity, torch.Tensor(torch.zeros(BATCH_SIZE)).cuda())
                d_loss = loss_real + loss_fake
                d_loss.backward()
                d_optim.step()


            """
            loss_G = E_{z ~ p(z)} [-log(D(G(z)))]
            """
            g_optim.zero_grad()
            noice = torch.Tensor(torch.randn(BATCH_SIZE, 100)).cuda()
            fake_labels = torch.Tensor(torch.LongTensor(np.random.randint(0, 10, BATCH_SIZE))).cuda()
            fake_images = G(noice, fake_labels)
            validity = D(fake_images, fake_labels)
            g_loss = loss_fn(validity, torch.Tensor(torch.ones(BATCH_SIZE)).cuda())
            g_loss.backward()
            g_optim.step()


    G = G.cpu()
    torch.save(G.parameters(), "generator.pt")
