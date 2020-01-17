import numpy as np
import scipy.io as sio
import torch
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from torchkbnufft import AdjMriSenseNufft, MriSenseNufft
from torchkbnufft.mri.mrisensesim import mrisensesim


def main():
    dtype = torch.double
    spokelength = 512
    targ_size = (int(spokelength/2), int(spokelength/2))
    nspokes = 405

    image = shepp_logan_phantom().astype(np.complex)
    im_size = image.shape
    grid_size = tuple(2 * np.array(im_size))

    # convert the phantom to a tensor and unsqueeze coil and batch dimension
    image = np.stack((np.real(image), np.imag(image)))
    image = torch.tensor(image).to(dtype).unsqueeze(0).unsqueeze(0)

    # create k-space trajectory
    ga = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))
    kx = np.zeros(shape=(spokelength, nspokes))
    ky = np.zeros(shape=(spokelength, nspokes))
    kmax = np.pi * ((spokelength/2) / im_size[0])
    ky[:, 0] = np.linspace(-kmax, kmax, spokelength)
    for i in range(1, nspokes):
        kx[:, i] = np.cos(ga) * kx[:, i - 1] - np.sin(ga) * ky[:, i - 1]
        ky[:, i] = np.sin(ga) * kx[:, i - 1] + np.cos(ga) * ky[:, i - 1]

    ky = np.transpose(ky)
    kx = np.transpose(kx)

    ktraj = np.stack((ky.flatten(), kx.flatten()), axis=0)
    ktraj = torch.tensor(ktraj).to(dtype).unsqueeze(0)

    # sensitivity maps
    ncoil = 8
    smap = np.absolute(np.stack(mrisensesim(
        im_size, coil_width=64))).astype(np.complex)
    smap = np.stack((np.real(smap), np.imag(smap)), axis=1)
    smap = torch.tensor(smap).to(dtype).unsqueeze(0)

    # operators
    sensenufft_ob = MriSenseNufft(
        smap=smap, im_size=im_size, grid_size=grid_size).to(dtype)

    kdata = sensenufft_ob(image, ktraj)

    kdata = np.squeeze(kdata.numpy())
    kdata = np.reshape(kdata[:, 0] + 1j*kdata[:, 1],
                       (ncoil, nspokes, spokelength))

    ktraj = np.squeeze(ktraj.numpy())
    ktraj = ktraj / np.max(ktraj) * np.pi
    ktraj = np.reshape(ktraj, (2, nspokes, spokelength))

    smap = np.squeeze(smap.numpy())
    smap = smap[:, 0] + 1j*smap[:, 1]
    smap_new = []
    for coilind in range(smap.shape[0]):
        smap_new.append(
            resize(np.real(smap[coilind]), targ_size) +
            1j*resize(np.imag(smap[coilind]), targ_size)
        )
    smap_new = np.array(smap_new)

    data = {
        'kdata': kdata,
        'ktraj': ktraj,
        'smap': smap_new
    }

    sio.savemat('demo_data.mat', data)


if __name__ == '__main__':
    main()
