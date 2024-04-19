## Tutorials and Examples

Prior to running any tutorials or examples, please complete the [Timeloop and
Accelergy tutorial
exercises](https://github.com/Accelergy-Project/timeloop-accelergy-exercises).

First, navigate to the `tutorials` directory and complete the tutorial Jupyter
notebooks. These tutorials will guide you through CiMLoop system specifications.
Note that the `demo_speed_accuracy.ipynb` notebook is not a tutorial to run, but
an artifact to show the speed and accuracy of the CiMLoop tool compared to prior
work.

After completing the tutorials, you may go explore example designs! The models
directory contains many examples of devices, circuits, architectures, and
workloads to play with. Navigate to the `models/arch/1_macro` directory and open
the `_guide.ipynb` file to play with the example designs.

Each `_guide.ipynb` file provides a guide to the macro in the given directory.
It also shows multiple experiments that both validate CiMLoop's model against
published results and explore the design space. When building your own
experiments, we recommend using these guides as a reference.

# File Guide

CiMLoop models are complex, as they allow for modification of every level of the
stack. To explore the design space, we recommend using an example design as a
starting point. We also recommend modifying only one level at first. To perform
more advanced explorations, you should familiarize yourself with all included
files and directories so you can modify them as needed.

### The `tutorials` Directory
The `tutorials` directory contains helpful tutorials to use the models.

- [Introduction](tutorials/1_cim_macro_intro.ipynb) 
- [Editing Variables](tutorials/2_editing_variables.ipynb) 
- [Components, Data Movement, and Reuse](tutorials/3_components_reuse.ipynb)
- More tutorials in progress!

### The `models` Directory

The models directory contains subdirectories covering the device, circuit,
architecture, and workload levels. The `top.yaml.jinja2` file gathers all the
files from the other points in this directory.

#### The `models/arch` directory: architecture
This directory is sub-divided into four levels: macro, tile, chip, and system.
Each of these may be composed to create a user-defined system. Users may mix and
match different options at each level to create different systems.

Of particular interest is the `1_macro` directory, which contains models of
published CiM macros. The `_guide.ipynb` file in each macro directory provides a
guide to the macro in the given directory.

All provided macros are CiM macros with the exception of
[albireo_isca_2021](models/arch/1_macro/albireo_isca_2021), which is a photonic
accelerator.

The `1_macro` directory contains several other important files:

- `_guide.ipynb`: A guide to the macro in the given directory. **Go here 
  first!** This guide includes validation and exploration experiments.
- `_tests.py`: The tests for the macro that are used in the guide. If you're
  creating new experiments, you can use these tests as a reference.
- `arch.yaml`: The architecture specification.
- `variables_iso.yaml`: Variables that should be kept constant when comparing
  different architectures for a fair comparison. These include data precision,
  technology node, and other parameters. Across different macros, we keep the
  same structure for the iso variables file such that we can mix and match
  different iso variables files and different macros.
- `variables_free.yaml`: Variables that do not need to be kept constant when
  comparing different architectures. These include clock frequency, numbers of
  components, and architecture-specific parameters.

#### The `models/components` directory: circuits and other components
This directory contains compound components and plug-ins that can be used in the
architecture.

#### The `models/memory_cells` directory: devices
This directory contains devices (memory cells) to be used in the architecture.

#### The `models/workloads` directory: workloads
This directory contains DNN workload layers.

#### The `models/include` directory: macros, defines, and helpful functions
This directory contains files that are included at various other points in the
system specification.

- defines.yaml: Includes various directives that can be used elsewhere.
- mapper.yaml: The Timeloop mapper settings.
- slicing_encoding.py: Contains functions for slicing and encoding operands.
- variables_common.yaml: Variables that are common across all architectures.

#### The `models/scripts` directory: supporting scripts
This directory contains scripts that make it easier to run models and visualize
results.
