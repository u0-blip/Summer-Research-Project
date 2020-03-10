from abaqus import *
from abaqusConstants import *
import __main__
import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior
from decimal import Decimal

import numpy as np

# working_dir = r'//ad.monash.edu/home/User045/dche145/Documents/Abaqus/microwave-break-rocks/'
working_dir = r'C:/peter_abaqus/Summer-Research-Project/'

pyrite_part = None
calcite_part = None
pyrite_ins = []
calcite_ins = []

part_pyrite_ins = []
part_calcite_ins = []
assem_pyrite_ins = []
assem_calcite_ins = []

part_name_file = working_dir + 'part_names.peter'
geo_distro_3D = working_dir + 'geometry.peter'
part_name_list = []
a_cells = []
b_cells = []

assem_name = 'Model-1'

model = mdb.models[assem_name]
print('in global var file')
print(model.name)
assembly = model.rootAssembly
num_change_flux = 25
magnitude = np.linspace(10, 500, num_change_flux)
timePeriod=50.0
increment=0.5
frame = int(np.ceil(timePeriod/increment))
num_intervals = 150

step = 1 
num_crystal = 0
a_size = [0.0, 0.0, 0.0]
b_size = [0.0, 0.0, 0.0]
theta_x = []
theta_y = []
theta_z = []
loc = []

new_session = True
clean_up_geo_test = False
from helpers import *

class part:
    def __init__(self, name, dim=[0.5, 0.5, 0.5], center=[0, 0, 0], shape='cube', b_create_part=False, assembly=assembly):
        self.assembly = assembly
        self.name = name
        self.dim = dim
        self.center = center
        self.shape = shape
        self.axis = [[self.dim[0], 0, 0], [0, self.dim[0], 0], [0, 0, self.dim[0]]]
        if self.shape == 'cube':
            s = model.ConstrainedSketch(
                name='__profile__', sheetSize=200.0)
            g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
            s.setPrimaryObject(option=STANDALONE)
            two_corner = ((-self.dim[0]/2+self.center[0], self.dim[1]/2+self.center[1]),
                (self.dim[0]/2 + self.center[0], -self.dim[1]/2 + self.center[1]))
            self.center[2] = self.dim[2]/2
            s.rectangle(point1=two_corner[0], point2=two_corner[1])
            p = model.Part(name=self.name,
                dimensionality=THREE_D, type=DEFORMABLE_BODY)
            p.BaseSolidExtrude(sketch=s, depth=self.dim[2])
            self.abq_part=p
            del model.sketches['__profile__']


class instance:
    def __init__(self, name, abq_part = None, script_part = None, center=(0,0,0)):
        self.name=name
        self.center = np.zeros(3)
        self.dim = np.zeros(3)
        if abq_part is not None:
            self.part = assembly.Instance(name=name, part=abq_part, dependent=ON)
            self.axis = [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]
        elif script_part is not None:
            self.dim = script_part.dim
            self.part = assembly.Instance(name=name, part=script_part.abq_part, dependent=ON)
            self.axis = script_part.axis
            self.translate((np.array(center) - np.array(script_part.center)).tolist())
            for i in range(3):
                self.center[i] = center[i]
        else:
            raise Exception('have to give one of the part')
        self.size = self.dim[0]*self.dim[1]*self.dim[2]

    def translate(self, vec):
        assembly.translate(instanceList=(self.name,), vector=vec)
        for i in range(3):
                self.center[i] = vec[i]

    def rotate(self, theta):
        rx = get_rx(theta[0])
        ry = get_ry(theta[1])
        rz = get_rz(theta[2])
        self.axis_temp = [[0,0,1] for i in range(3)]
        assembly.rotate(instanceList=(self.name,), axisPoint=[0,0,0],
        axisDirection=self.axis[0], angle=theta[0]*180/np.pi)
        self.axis_temp[1] = matmul_vec(rx, matmul_vec(ry, self.axis[1]))
        assembly.rotate(instanceList=(self.name,), axisPoint=[0,0,0],
        axisDirection=self.axis_temp[1], angle=theta[1]*180/np.pi)
        self.axis_temp[2] = matmul_vec(rx, matmul_vec(ry, matmul_vec(rz, self.axis[2])))
        assembly.rotate(instanceList=(self.name,), axisPoint=[0,0,0],
        axisDirection=self.axis_temp[2], angle=theta[2]*180/np.pi)

        # for i in range(3):
        #     self.axis_temp[i] = matmul_vec(rx, matmul_vec(ry, matmul_vec(rz, self.axis[i])))
        #     assembly.DatumPointByCoordinate(self.axis_temp[i])
        # if clean_up_geo_test:
        #     del assembly.features['ax_0']
        #     del assembly.features['ax_1']
        #     del assembly.features['ax_2']
        # assembly.features.changeKey(
        #     fromName='Datum pt-1', toName='ax_0')
        # assembly.features.changeKey(
        #     fromName='Datum pt-2', toName='ax_1')
        # assembly.features.changeKey(
        #     fromName='Datum pt-3', toName='ax_2')

