import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Check for GPU availability and set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

MODEL_SAVE_PATH = 'saved_models/'
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


def save_generated_images(epoch, generator, noise, path='generated_images/'):
    if not os.path.exists(path):
        os.makedirs(path)
    with torch.no_grad():
        fake = generator(noise).detach().cpu()
    
    # Save the grid of images (e.g., 8x8 grid)
    grid_image = vutils.make_grid(fake, padding=2, normalize=True, nrow=8) # Assuming 8x8 grid
    vutils.save_image(grid_image, f"{path}/grid_epoch_{epoch}.png")
    
    # Save the first generated image from the batch as a single image
    vutils.save_image(fake[0], f"{path}/single_epoch_{epoch}.png", normalize=True)

def save_model_weights(epoch, generator, discriminator, path=MODEL_SAVE_PATH):
    torch.save(generator.state_dict(), os.path.join(path, f"generator_epoch_{epoch}.pth"))
    torch.save(discriminator.state_dict(), os.path.join(path, f"discriminator_epoch_{epoch}.pth"))
    logging.info(f"Saved model weights for epoch {epoch}")

# Hyperparameters
NUM_EPOCHS = 1000
BATCH_SIZE = 64
LR = 0.0002
BETA1 = 0.5
LATENT_DIM = 100

# Dataset & DataLoader
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = datasets.ImageFolder(root='./Images', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Create Generator and Discriminator
generator = Generator(LATENT_DIM).to(device)
discriminator = Discriminator().to(device)

# Loss and Optimizers
criterion = nn.BCELoss().to(device)
optimizer_g = optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))

# Define a fixed noise vector to monitor the progress of the Generator
fixed_noise = torch.randn(64, LATENT_DIM, 1, 1).to(device)

# Training Loop
for epoch in range(NUM_EPOCHS):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        current_batch_size = real_images.size(0)
        real_labels = torch.ones(current_batch_size, 1).to(device)
        fake_labels = torch.zeros(current_batch_size, 1).to(device)
        
        # Discriminator update
        optimizer_d.zero_grad()

        # Real images
        outputs = discriminator(real_images).view(-1, 1)
        d_loss_real = criterion(outputs, real_labels)
        d_loss_real.backward()

        # Fake images
        noise = torch.randn(current_batch_size, LATENT_DIM, 1, 1)
        fake_images = generator(noise)
        outputs = discriminator(fake_images.detach()).view(-1, 1)
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss_fake.backward()

        d_loss = d_loss_real + d_loss_fake
        optimizer_d.step()

        # Generator update
        optimizer_g.zero_grad()
        outputs = discriminator(fake_images).view(-1, 1)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_g.step()

    logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS} - Discriminator Loss: {d_loss:.4f}, Generator Loss: {g_loss:.4f}")

    if (epoch + 1) % 10 == 0:
        save_generated_images(epoch + 1, generator, fixed_noise)

logging.info("Training completed!")
