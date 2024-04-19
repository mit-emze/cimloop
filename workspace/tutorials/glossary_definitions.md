wordline:
  A physical wire that runs horizontally across a memory array.

bitline:
  A physical wire that runs vertically across a memory array.

row:
  A port via which one operand can be communicated to/from wordline(s) in
  a given timestep. Note that one row can be connected to multiple wordlines
  if the memory array is partitioned into multiple sub-arrays that are
  accessed one at a time.
  
column:
  A port via which one operand can be communicated to/from bitline(s) in
  a given timestep. Note that one column can be connected to multiple bitlines
  if the memory array. This often occurs if the outputs generated from
  multiple adjacent memory cells are summed together in a single column.

cim_unit:
  The intersection of a row and a column. This is the smallest unit of
  computation in a CiM array, and can perform one mac in a given
  timestep.

memory_cell:
  A single memory cell in a CiM array. This may be an SRAM bitcell, an RRAM
  device, a DRAM bitcell, an RRAM device with an access transistor, etc.

cim_unit_width_cells:
  The number of adjacent memory cells that together store different bits of
  one operand, and whose outputs are summed together in a single column.

cim_unit_depth_cells:
  The number of adjacent memory cells that together store different bits of
  one operand, and who are connected to the same row. Different-depth cells
  must be accessed in different timesteps.

timestep:
  For a given component, a timestep is one iteration of the lowest-level
  temporal loop that is above that component. Note that temporal loops beneath
  the component may iterate multiple times within a timestep for that component,
  and as such, the duration of a timestep increases monotonicly as we move
  up the hierarchy (away from compute and toward main memory).