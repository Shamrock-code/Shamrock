import numpy as np
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import sys
import pandas as pd


def _bin_particles_by_radius(data: pd.Series,
                             r_in: float = None,
                             r_out: float = None,
                             bins: int = 5,
                             geometry: str = 'cylindrical',
                             origin: list = None):


    if geometry == 'spherical':
        r = np.sqrt(    (data['x'] - origin[0]) ** 2
                    +   (data['y'] - origin[1]) ** 2
                    +   (data['z'] - origin[2]) ** 2)
    elif geometry == 'cylindrical':
        r = np.sqrt(    (data['x'] - origin[0]) ** 2
                    +   (data['y'] - origin[1]) ** 2)
    else:
        raise ValueError("geometry should be either 'cylindrical' or 'spherical'")

    # should we add epsilon here?
    if r_in is None:
        r_in = r.min() - sys.float_info.epsilon
    if r_out is None:
        r_out = r.max() + sys.float_info.epsilon

    bin_edges = np.linspace(r_in, r_out, bins+1)
    rbins = pd.cut(r, bin_edges)

    return rbins, bin_edges



def _get_bin_midpoints(bin_edges: np.ndarray) -> np.ndarray:
    """
    Calculate the midpoint of bins given their edges.
    """

    return 0.5 * (bin_edges[1:] - bin_edges[:-1]) + bin_edges[:-1]

def _calc_angular_momentum(data: pd.Series,
                           rbins: pd.Series,
                           origin: list,
                           unit_vector: bool,
                           mass: float):

    x_data = data['x'] - origin[0]
    print("################# X_DATA TYPE {}".format(type(x_data)))
    y_data = data['y'] - origin[1]
    z_data = data['z'] - origin[2]

    Lx = (y_data * data['vz'] - z_data * data['vy']) * mass
    Ly = (z_data * data['vx'] - x_data * data['vz']) * mass
    Lz = (x_data * data['vy'] - y_data * data['vx']) * mass

    Lx = pd.DataFrame({'Lx': Lx})
    Ly = pd.DataFrame({'Lx': Ly})
    Lz = pd.DataFrame({'Lx': Lz})

    #L_pd = pd.DataFrame({'Lx': Lx, 'Ly': Ly, 'Lz': Lz})
    print("################# Lx TYPE {}".format(type(Lx)))

    #L_pd['r'] = r
    #L_pd['rbins'] = rbins

    if isinstance(mass, float):
        #L_pd['Lx'] = L_pd.groupby('rbins')['Lx'].sum()
        #L_pd['Ly'] = L_pd.groupby('rbins')['Ly'].sum()
        #L_pd['Lz'] = L_pd.groupby('rbins')['Lz'].sum()

        Lx = (mass * Lx).groupby(rbins).sum()
        Ly = (mass * Ly).groupby(rbins).sum()
        Lz = (mass * Lz).groupby(rbins).sum()

    else:
        raise ValueError("mass is not a float!")

    if unit_vector:
        Lmag = 1.0 / np.sqrt(Lx ** 2 + Ly ** 2 + Lz ** 2)

        Lx = Lx * Lmag
        Ly = Ly * Lmag
        Lz = Lz * Lmag

    return Lx, Ly, Lz


def angular_momentum(data: pd.Series,
                     r_in: float = None,
                     r_out: float = None,
                     bins: int = 300,
                     geometry: str = 'cylindrical',
                     origin: list = None,
                     retbins: bool = False,
                     unit_vector: bool = True):
  

    rbins, bin_edges = _bin_particles_by_radius(data, r_in, r_out, bins,
                                                geometry, origin)

    Lx, Ly, Lz = _calc_angular_momentum(data, rbins, origin= [0,0,0], unit_vector=unit_vector, mass=0.001)

    if retbins:
        return Lx, Ly, Lz, _get_bin_midpoints(bin_edges)
    else:
        return Lx, Ly, Lz
    



def read_vtk(filename):
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(filename)
    reader.Update()



    # Get the output data from the reader
    vtk_data = reader.GetOutput()

    # Get the number of arrays in the point data
    num_arrays = vtk_data.GetPointData().GetNumberOfArrays()

    # Create a dictionary to store column names and data
    columns_dict = {}

    # Iterate through each array and extract data
    for i in range(num_arrays):
        vtk_array = vtk_data.GetPointData().GetArray(i)
        array_name = vtk_array.GetName()

        # Convert VTK array to NumPy array
        numpy_array = vtk_to_numpy(vtk_array)

        # Store the column name and data in the dictionary
        columns_dict[array_name] = numpy_array

    # Access the spatial positions (coordinates) of the points
    points = vtk_data.GetPoints()
    numpy_points = vtk_to_numpy(points.GetData())

    columns_dict['x'] = numpy_points[:, 0]
    columns_dict['y'] = numpy_points[:, 1]
    columns_dict['z'] = numpy_points[:, 2]

    columns_dict['vx'] = columns_dict['v'][:, 0]
    columns_dict['vy'] = columns_dict['v'][:, 1]
    columns_dict['vz'] = columns_dict['v'][:, 2]

    # Print or use the extracted columns as needed
    #for column_name, column_data in columns_dict.items():
    #    print(f"{column_name}: {column_data}")

    series = pd.Series(columns_dict)
    return series



file = "/home/ylapeyre/Shamrock_tests/dump.vtk"
mydata = read_vtk(file)

#rbins = _bin_particles_by_radius(mydata, origin=[0,0,0])
#Lx, Ly, Lz = _calc_angular_momentum(mydata, mass=0.001, origin=[0,0,0], unit_vector=False)
#print(L_pd['Lx'].shape)


Lx, Ly, Lz, bin_mid = angular_momentum(mydata, origin=[0,0,0], unit_vector=True, retbins=True)
twist = np.arctan(Lx / Lz) #ly / lx in Nealon 2015
tilt = np.arccos(Ly)



fig, ax = plt.subplots(1,5,figsize=(5,4))
#ax.plot(unit_am[3],unit_am[0], label='unit x')

ax[0].plot(bin_mid, Lx, label='Lx', c='g')
ax[0].plot(bin_mid, Ly, label='Ly', c='r')
ax[0].plot(bin_mid, Lz, label='Lz', c='b')
ax[0].set_xlabel('Radius')
ax[0].set_title('Unit angular momenta')
ax[0].legend()

ax[1].scatter(bin_mid, tilt, c='k', marker='.')
ax[1].set_title('tilt')

ax[2].scatter(bin_mid, twist, c='k', marker='.')
ax[2].set_title('twist')

ax[3].scatter(mydata['x'], mydata['x'], label='x', c='g', marker='.', alpha=0.5)
#ax[3].scatter(mydata['x'], mydata['y'], label='y', c='r', marker='.')
ax[3].scatter(mydata['x'], mydata['z'], label='z', c='b', marker='.', alpha=0.5)
ax[3].legend()

ax[4].scatter(mydata['x'], mydata['vx'], label='vx', c='g', marker='.', alpha=0.5)
#ax[4].scatter(mydata['x'], mydata['vy'], label='vy', c='r', marker='.')
ax[4].scatter(mydata['x'], mydata['vz'], label='vz', c='b', marker='.', alpha=0.5)
ax[4].legend()


plt.show()
