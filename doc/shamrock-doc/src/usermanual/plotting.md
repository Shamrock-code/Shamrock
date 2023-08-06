# SPH

```py
plt.figure(dpi=200)
import copy
my_cmap = copy.copy(matplotlib.colormaps.get_cmap('plasma')) # copy the default cmap
my_cmap.set_bad(my_cmap.colors[0])
sinks = np.array( model.get_sinks_pos())
print(sinks)
arr = model.compute_slice("rho",min_coord = (-2.,0,-2.),delta_x = (4.,0,0.),delta_y = (0.,0,4.), nx = 1000, ny = 1000)
res = plt.imshow(arr, cmap=my_cmap,origin='lower', extent=[-2, 2, -2, 2], norm="log", vmin=1e-2, vmax=1e3)
plt.scatter(sinks[:,0],sinks[:,2])
plt.xlabel("x")
plt.ylabel("y")
plt.title("t = {:0.3f}s".format(t_sum))

cbar = plt.colorbar(res, extend='both')
cbar.set_label('$rho$')
#plt.show()
plt.savefig("plots/plot_{:04}.png".format(i_dump))
```