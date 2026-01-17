
import numpy
import matplotlib.pyplot as plt



def TurbField(res=256, minmode=2, maxmode=64, sol_weight=1.0, seed=42):
    from scipy import  fftpack

    freqs = fftpack.fftfreq(res)
    freq3d = numpy.array(numpy.meshgrid(freqs, freqs, freqs, indexing="ij"))
    intfreq = numpy.around(freq3d * res)
    kSqr = numpy.sum(numpy.abs(freq3d) ** 2, axis=0)
    intkSqr = numpy.sum(numpy.abs(intfreq) ** 2, axis=0)
    VK = []

    # apply ~k^-2 exp(-k^2/kmax^2) filter to white noise to get x, y, and z components of velocity field
    for i in range(3):
        numpy.random.seed(seed + i)
        rand_phase = fftpack.fftn(
            numpy.random.normal(size=kSqr.shape)
        )  # fourier transform of white noise
        vk = rand_phase * (float(minmode) / res) ** 2 / (kSqr + 1e-300)
        vk[intkSqr == 0] = 0.0
        vk[intkSqr < minmode**2] *= (
            intkSqr[intkSqr < minmode**2] ** 2 / minmode**4
        )  # smoother filter than mode-freezing; should give less "ringing" artifacts
        vk *= numpy.exp(-intkSqr / maxmode**2)

        VK.append(vk)
    VK = numpy.array(VK)
    # bin = numpy.logspace(0,2.5,50)
    # plt.hist(vk.flatten(),bins=bin)
    # #plt.xlim(0,10**2.)
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.show()
    # plt.imshow(vk[:,25,:].real)
    # plt.show()
    vk_new = numpy.zeros_like(VK)

    # do projection operator to get the correct mix of compressive and solenoidal
    for i in range(3):
        for j in range(3):
            if i == j:
                vk_new[i] += sol_weight * VK[j]
            vk_new[i] += (
                (1 - 2 * sol_weight) * freq3d[i] * freq3d[j] / (kSqr + 1e-300) * VK[j]
            )
    vk_new[:, kSqr == 0] = 0.0
    VK = vk_new

    vel = numpy.array(
        [fftpack.ifftn(vk).real for vk in VK]
    )  # transform back to real space
    vel -= numpy.average(vel, axis=(1, 2, 3))[:, numpy.newaxis, numpy.newaxis, numpy.newaxis]
    vel = vel / numpy.sqrt(numpy.sum(vel**2, axis=0).mean())  # normalize so that RMS is 1
    return numpy.array(vel,dtype='f4')


# Global variables for velocity field
vx_global = None
vy_global = None
vz_global = None
domain_size_global = 1.0

def vel_field(pos):
    """
    Interpolate velocity at position (x, y, z) using global velocity fields.
    
    Parameters:
    -----------
    pos : tuple
        (x, y, z) position at which to interpolate the velocity
    
    Returns:
    --------
    tuple
        (vx, vy, vz) velocity components at the given position(s)
    """
    from scipy.interpolate import RegularGridInterpolator
    
    global vx_global, vy_global, vz_global, domain_size_global
    
    x, y, z = pos
    res = vx_global.shape[0]
    
    # Create coordinate arrays for the grid
    coords = numpy.linspace(0, domain_size_global, res)
    
    # Create interpolators for each velocity component
    interp_vx = RegularGridInterpolator((coords, coords, coords), vx_global, 
                                         bounds_error=False, fill_value=0.0)
    interp_vy = RegularGridInterpolator((coords, coords, coords), vy_global, 
                                         bounds_error=False, fill_value=0.0)
    interp_vz = RegularGridInterpolator((coords, coords, coords), vz_global, 
                                         bounds_error=False, fill_value=0.0)
    
    points = numpy.column_stack([numpy.atleast_1d(x), 
                                  numpy.atleast_1d(y), 
                                  numpy.atleast_1d(z)])
    
    vx_interp = interp_vx(points)[0]
    vy_interp = interp_vy(points)[0]
    vz_interp = interp_vz(points)[0]
    
    return vx_interp, vy_interp, vz_interp


if __name__ == "__main__":

    seed = 42

    #avx,avy,avz = make_turb_field(res,power,seed)
    vx,vy,vz = TurbField(128,2,64,0.7,seed)

    print(vx)
    print(vy)
    print(vz)
    
    # Set global velocity fields
    vx_global = vx
    vy_global = vy
    vz_global = vz
    domain_size_global = 1.0
    
    # Example: Interpolate velocity at a specific position
    test_pos = (0.5, 0.5, 0.5)
    vel_x, vel_y, vel_z = vel_field(test_pos)
    print(f"\nVelocity at {test_pos}:")
    print(f"  vx = {vel_x:.6f}")
    print(f"  vy = {vel_y:.6f}")
    print(f"  vz = {vel_z:.6f}")
