"""Assignment 9
Part 2: AC-GAN

NOTE: Feel free to check: https://arxiv.org/pdf/1610.09585.pdf

NOTE: Write Down Your Info below:

    Name: Vedd Kuknur

    CCID: kuknur

    Auxiliary Test Accuracy on Cifar10 Test Set:


"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

def compute_score(acc, min_thres, max_thres):
    if acc <= min_thres:
        base_score = 0.0
    elif acc >= max_thres:
        base_score = 100.0
    else:
        base_score = float(acc - min_thres) / (max_thres - min_thres) \
                     * 100
    return base_score


# -----
# AC-GAN Build Blocks

# #####
# TODO: Complete the generator architecture
# #####

class Generator(nn.Module):
    def __init__(self, latent_dim=128, out_channels=3, num_classes=10):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.out_channels = out_channels

        # #####
        # TODO: Complete the generator architecture
        # #####
        self.label_emb = nn.Embedding(self.num_classes, self.latent_dim)
        self.gen_lin = nn.Linear(latent_dim, 2048)
        self.gen = nn.Sequential(
                    nn.ConvTranspose2d(128,64,kernel_size=3,stride = 2,padding=1,output_padding=1),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(64,32,kernel_size=3,stride = 2,padding=1,output_padding=1),
                    nn.BatchNorm2d(32),
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(32,32,kernel_size=3,stride=2,padding=1,output_padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(32, 3,kernel_size= 3, padding= 1),
                    nn.Sigmoid()
                    )

        
    def forward(self, z, y):
        # #####
        # TODO: Complete the generator architecture
        # #####
        gen_input = torch.mul(self.label_emb(y), z)
        z = self.gen_lin(gen_input)  
        z = z.view(-1, 128, 2, 2)
        z = self.gen(z)
        z = z.view(-1, 3, 32, 32)
        return z

# #####
# TODO: Complete the Discriminator architecture
# #####

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        # #####
        # TODO: Complete the discriminator architecture
        # #####
        self.dis = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.adv_layer = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(4, 10), nn.Softmax())


    def forward(self, x):
        # #####
        # TODO: Complete the discriminator architecture
        # #####
        x=self.dis(x)
        x_adv=self.adv_layer(x)
        x_aux= self.aux_layer(x)
        return x_adv,x_aux
        


# -----
# Hyperparameters
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# NOTE: Feel free to change the hyperparameters as long as you meet the marking requirement
batch_size = 128
workers = 6
latent_dim = 128
lr = 0.001
num_epochs = 150
validate_every = 1
print_every = 100

save_path = os.path.join(os.path.curdir, "visualize", "gan")
if not os.path.exists(os.path.join(os.path.curdir, "visualize", "gan")):
    os.makedirs(os.path.join(os.path.curdir, "visualize", "gan"))
ckpt_path = 'acgan.pt'

# -----
# Dataset
# NOTE: Data is only normalized to [0, 1]. THIS IS IMPORTANT!!!
tfms = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=tfms)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=tfms)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=workers)

# -----
# Model
# #####
# Initialize your models HERE.
# #####
generator = Generator()

discriminator = Discriminator()

# -----
# Losses

# #####
# Initialize your loss criterion.
# #####

adv_loss = nn.BCELoss()
aux_loss = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    adv_loss = adv_loss.cuda()
    aux_loss = aux_loss.cuda()

# Optimizers for Discriminator and Generator, separate

# #####
# Initialize your optimizer(s).
# #####

optimizer_D = torch.optim.Adam(generator.parameters(), lr=lr)
optimizer_G = torch.optim.Adam(discriminator.parameters(), lr=lr)

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


# -----
# Train loop

def denormalize(x):
    """Denomalize a normalized image back to uint8.
    """
    # #####
    # Complete denormalization.
    # #####
    x = x.cpu()
    return torch.round(x * 255).int()


# For visualization part
# Generate 20 random sample for visualization
# Keep this outside the loop so we will generate near identical images with the same latent featuresper train epoch
random_z = None
random_y = None


# #####
# TODO: Complete train_step for AC-GAN
# #####

def train_step(x, y):
    global random_y,random_z
    """One train step for AC-GAN.
    You should return loss_g, loss_d, acc_d, a.k.a:
        - average train loss over batch for generator
        - average train loss over batch for discriminator
        - auxiliary train accuracy over batch
    """
    batch_size = x.shape[0]

    # Adversarial ground truths
    valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
    fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

    # Configure input
    real_imgs = Variable(x.type(FloatTensor))
    labels = Variable(y.type(LongTensor))

    # -----------------
    #  Train Generator
    # -----------------

    optimizer_G.zero_grad()

    # Sample noise and labels as generator input
    z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
    gen_labels = Variable(LongTensor(np.random.randint(0, len(classes), batch_size)))

    # Generate a batch of images
    gen_imgs = generator(z, gen_labels)

    # Loss measures generator's ability to fool the discriminator
    validity, pred_label = discriminator(gen_imgs)
    g_loss = 0.5 * (adv_loss(validity, valid) + aux_loss(pred_label, gen_labels))

    g_loss.backward()
    optimizer_G.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------

    optimizer_D.zero_grad()

    # Loss for real images
    real_pred, real_aux = discriminator(real_imgs)
    d_real_loss = (adv_loss(real_pred, valid) + aux_loss(real_aux, labels)) / 2

    # Loss for fake images
    fake_pred, fake_aux = discriminator(gen_imgs.detach())
    d_fake_loss = (adv_loss(fake_pred, fake) + aux_loss(fake_aux, gen_labels)) / 2

    # Total discriminator loss
    d_loss = (d_real_loss + d_fake_loss) / 2

    # Calculate discriminator accuracy
    pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
    gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
    d_acc = np.mean(np.argmax(pred, axis=1) == gt)

    d_loss.backward()
    optimizer_D.step()

    random_z = Variable(FloatTensor(np.random.normal(0, 1, (10 ** 2, latent_dim))))
    random_y = np.array([num for _ in range(10) for num in range(10)])
    random_y = Variable(LongTensor(random_y))
    return g_loss, d_loss, d_acc

def test(
        test_loader,
):
    """Calculate accuracy over Cifar10 test set.
    """
    size = len(test_loader.dataset)
    corrects = 0

    discriminator.eval()
    with torch.no_grad():
        for inputs, gts in test_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                gts = gts.cuda()

            # Forward only
            _, outputs = discriminator(inputs)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == gts.data)

    acc = corrects / size
    print("Test Acc: {:.4f}".format(acc))
    return acc


g_losses = []
d_losses = []
best_acc_test = 0.0

for epoch in range(1, num_epochs + 1):
    generator.train()
    discriminator.train()

    avg_loss_g, avg_loss_d = 0.0, 0.0
    for i, (x, y) in enumerate(train_loader):
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        # train step
        loss_g, loss_d, acc_d = train_step(x, y)
        avg_loss_g += loss_g * x.shape[0]
        avg_loss_d += loss_d * x.shape[0]

        # Print
        if i % print_every == 0:
            print(
                "Epoch {}, Iter {}: LossD: {:.6f} LossG: {:.6f}, D_acc {:.6f}".format(epoch, i, loss_g, loss_d, acc_d))

    g_losses.append(avg_loss_g / len(train_dataset))
    d_losses.append(avg_loss_d / len(train_dataset))

    # Save
    if epoch % validate_every == 0:
        acc_test = test(test_loader)
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            # Wrap things to a single dict to train multiple model weights
            state_dict = {
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
            }
            torch.save(state_dict, ckpt_path)
            print("Best model saved w/ Test Acc of {:.6f}.".format(best_acc_test))

        # Do some reconstruction
        generator.eval()
        with torch.no_grad():
            # Forward
            xg = generator(random_z, random_y)
            xg = denormalize(xg)

        #     # Plot 20 randomly generated images
        #     plt.figure(figsize=(10, 5))
        #     for p in range(20):
        #         plt.subplot(4, 5, p + 1)
        #         plt.imshow(xg[p])
        #         plt.text(0, 0, "{}".format(classes[random_y[p].item()]), color='black',
        #                  backgroundcolor='white', fontsize=8)
        #         plt.axis('off')
        #
        #     plt.savefig(os.path.join(os.path.join(save_path, "E{:d}.png".format(epoch))), dpi=300)
        #     plt.clf()
        #     plt.close('all')
        #
        # # Plot losses
        # plt.figure(figsize=(10, 5))
        # plt.title("Generator and Discriminator Loss During Training")
        # plt.plot(g_losses, label="G")
        # plt.plot(d_losses, label="D")
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.xlim([1, epoch])
        # plt.legend()
        # plt.savefig(os.path.join(os.path.join(save_path, "loss.png")), dpi=300)

# Just for you to check your Part 2 score
score = compute_score(best_acc_test, 0.65, 0.69)
print("Your final accuracy:", best_acc_test)
print("Your Assignment Score:", score)
