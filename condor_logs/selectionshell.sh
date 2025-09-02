#!/bin/sh



#qsub -v name="TCNGumbelRegHD1",M=1,seed=1,epochs=200 selectionscript.pbs
qsub -v name="TCNGumbelRegHD1",M=2,seed=1,epochs=200 selectionscript.pbs
qsub -v name="TCNGumbelRegHD1",M=3,seed=1,epochs=200 selectionscript.pbs
qsub -v name="TCNGumbelRegHD1",M=4,seed=1,epochs=200 selectionscript.pbs
qsub -v name="TCNGumbelRegHD1",M=6,seed=1,epochs=200 selectionscript.pbs
qsub -v name="TCNGumbelRegHD1",M=8,seed=1,epochs=200 selectionscript.pbs
qsub -v name="TCNGumbelRegHD1",M=10,seed=1,epochs=200 selectionscript.pbs

qsub -v name="TCNGumbelRegHD2",M=1,seed=2,epochs=300 selectionscript.pbs
qsub -v name="TCNGumbelRegHD2",M=2,seed=2,epochs=300 selectionscript.pbs
qsub -v name="TCNGumbelRegHD2",M=3,seed=2,epochs=300 selectionscript.pbs
qsub -v name="TCNGumbelRegHD2",M=4,seed=2,epochs=300 selectionscript.pbs
qsub -v name="TCNGumbelRegHD2",M=6,seed=2,epochs=300 selectionscript.pbs
qsub -v name="TCNGumbelRegHD2",M=8,seed=2,epochs=300 selectionscript.pbs
qsub -v name="TCNGumbelRegHD2",M=10,seed=2,epochs=300 selectionscript.pbs

qsub -v name="TCNGumbelRegHD3",M=1,seed=3,epochs=400 selectionscript.pbs
qsub -v name="TCNGumbelRegHD3",M=2,seed=3,epochs=400 selectionscript.pbs
qsub -v name="TCNGumbelRegHD3",M=3,seed=3,epochs=400 selectionscript.pbs
qsub -v name="TCNGumbelRegHD3",M=4,seed=3,epochs=400 selectionscript.pbs
qsub -v name="TCNGumbelRegHD3",M=6,seed=3,epochs=400 selectionscript.pbs
qsub -v name="TCNGumbelRegHD3",M=8,seed=3,epochs=400 selectionscript.pbs
qsub -v name="TCNGumbelRegHD3",M=10,seed=3,epochs=400 selectionscript.pbs
