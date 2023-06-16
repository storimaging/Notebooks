import torch

# Code based on DCGANs tutorial https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
def train(train_loader, optimizerD, optimizerG, y_1, y_0, criterion, num_epochs, log_every):
    zviz = torch.randn(batch_size,nz,1,1).to(device)
    for epoch in range(num_epochs):
        # For each batch in the train_loader
        for i, batch in enumerate(train_loader, 0):

            ############################
            # Batchs of real and fake images
            real = batch[0].to(device)
            fake = netG(torch.randn(batch_size, nz, 1, 1, device=device))
            
            ############################
            # Update D network
            netD.zero_grad()

            # Forward pass real batch. Calculate loss. Calculate gradients
            output = netD(real)
            errD_real = criterion(output, y_1) 
            errD_real.backward()

            # Forward pass fake batch. Calculate loss. Calculate gradients
            output = netD(fake.detach())
            errD_fake = criterion(output, y_0) 
            errD_fake.backward()

            # Compute error of D. Update D
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # Update G network
            netG.zero_grad()

            # Forward pass fake batch. Calculate loss. Calculate gradients
            output = netD(fake)
            errG = criterion(output, y_1)
            errG.backward()

            # Update G
            optimizerG.step()
            
            ############################
            # Display training stats and visualize
            if i % log_every == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f' % (epoch, num_epochs, i, len(train_loader), errD.item(), errG.item()))
                show_netG(zviz)
                

def InterpolationInLatentSpace():
    with torch.no_grad():
        nlatent = 10
        ninterp = 20
        z0 = torch.randn(nlatent,k).to(device)
        z1 = torch.randn(nlatent,k).to(device)

        alpha = torch.linspace(0.,1.,ninterp).view(ninterp,1,1).to(device)

        z = alpha*z0 + (1.-alpha)*z1
        z = z.transpose(1,0)
        print(z.size())
        genimages = G_net(z)
        imshow(torchvision.utils.make_grid(genimages.to('cpu'), nrow=ninterp))

