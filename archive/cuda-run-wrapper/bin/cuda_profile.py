#!/usr/bin/python
import os
import sys
import time
import random

run_script = 'bin/cuda_run.py'
program = sys.argv[1]
intervel = 0.4
n_experiment = 100

class Profiler:
    def __init__(self, run_script, program, intervel, n_experiment):
        self.run_script = run_script
        self.program = program
        self.intervel = intervel
        self.n_experiment = n_experiment
             
    def run(self):
        for i in range(self.n_experiment):
            self.experiment()
            time.sleep(random.random()*self.intervel)

    def experiment(self):
        os.system('%s run %s &' %(self.run_script, self.program))
        # os.system('%s&'%self.program)

    def analyze(self):
        pass

profiler = Profiler(run_script, program, intervel, n_experiment)    
profiler.run()
profiler.analyze()
        
             
             
             
    
