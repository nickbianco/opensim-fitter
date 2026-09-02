import os
import opensim as osim

model = osim.Model(os.path.join('results', 'subject_bilevel_scaled_athlete.osim'))
model.initSystem()

table = osim.TimeSeriesTable(
    os.path.join('results', 'boxjump_1_1_bilevel_solution.sto'))
table.addTableMetaDataString('inDegrees', 'no')

viz = osim.VisualizerUtilities()
viz.showMotion(model, table)