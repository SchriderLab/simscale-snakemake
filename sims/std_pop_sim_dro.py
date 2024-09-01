import stdpopsim
import numpy as np
from contextlib import redirect_stdout

species = stdpopsim.get_species('DroMel')
model = stdpopsim.PiecewiseConstantSize(species.population_size)
contig = species.get_contig(length=1e4)
dfe = species.get_dfe('LognormalPlusPositive_R16')
contig.add_dfe(intervals=np.array([[0, int(contig.length)]]), DFE=dfe)
samples = {'pop_0': 0}
engine = stdpopsim.get_engine("slim")
with open('dro_scaled_stdpopsim.slim', 'w+') as f:
    with redirect_stdout(f):
        ts = engine.simulate(model, contig, samples, slim_scaling_factor=20, slim_script=True)
