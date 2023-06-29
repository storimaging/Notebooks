import torch
import torchvision

# Code based on DCGANs tutorial https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
def train(train_loader, optimizerD, optimizerG, y_1, y_0, criterion, num_epochs, log_every, batch_size, nz, device, netG,netD, show_netG):
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
                

def InterpolationInLatentSpace(batch_size, nz, device, netG, imshow):
    nlatent = 10
    ninterp = 20
    genimages = torch.empty(0,1,28,28).to(device)

    with torch.no_grad():
        z0 = torch.randn(batch_size, nz, 1, 1, device=device)
        z1 = torch.randn(batch_size, nz, 1, 1, device=device) 
        alpha = torch.linspace(0.,1.,ninterp).to(device) 

        for a in alpha:
           z = a*z0 + (1.-a)*z1 
           imgs = netG(z)[:nlatent] # returns 128 images from the generator model. Save the first #nlatent
           genimages = torch.cat((genimages, imgs), dim=0) # add images to result vector for later display
        # transpose the order of images and display:   
        genimages = genimages.reshape(nlatent, ninterp,1,28,28)
        genimages = genimages.transpose(0,1)
        genimages = genimages.reshape(nlatent*ninterp,1,28,28)
        imshow(torchvision.utils.make_grid(genimages.to('cpu'), nrow=ninterp))


