.. index:: Ipopt plugin overview

Overview of the Ipopt Plugin
==============================

The Ipopt Plugin is a wrapper for the 
`Ipopt optimizer
<http://www.coin-or.org/Ipopt/>`_. Ipopt (Interior Point OPTimizer) 
is a software package for large-scale nonlinear optimization.

The rest of this document assumes that you have already installed OpenMDAO and understand
some of the basics of using OpenMDAO.
This document also assumes that the user has some understanding of how Ipopt
works. In particular, the user should aware of all the 
`options for Ipopt
<http://www.coin-or.org/Ipopt/documentation/node59.html#app.options_ref>`_.

In addition, there is a 
`short Ipopt tutorial is available
<http://drops.dagstuhl.de/volltexte/2009/2089/pdf/09061.WaechterAndreas.Paper.2089.pdf>`_ 
and a 
`longer tutorial
<https://projects.coin-or.org/Ipopt/export/2054/stable/3.9/Ipopt/doc/documentation.pdf>`_. 


.. note::  In addition to the requirement of having Ipopt installed, 
           the Python wrapper,
           `Pyipopt 
           <http://code.google.com/p/pyipopt/>`_ needs to be installed.

.. note::  Ipopt is built using a variety of third party libraries for 
           solving equations. The OpenMDAO Ipopt driver
           was tested using the following Ipopt third party libraries:
           BLAS, LAPACK, MUMPS, ASL and Metis.


How Do I Use the Ipopt Plugin?
-------------------------------------

Using the plugin is like using other optimizer drivers available in 
OpenMDAO. 

Here is some example code. The comments explain some details of the usage of this
component.

.. testcode:: Ipoptdriver

    import numpy
    
    from openmdao.main.api import Assembly, Component, set_as_top
    from openmdao.lib.datatypes.api import Array, Float
    
    from ipoptdriver.ipoptdriver import IPOPTdriver
    
    
    class ParaboloidComponent(Component):
        """     
             MINIMIZE OBJ = ( X(1) - 2.0 ) ** 2 +  ( X(2) - 3.0 ) **2
        """
        
        x = Array(iotype='in', low=-1e99, high=1e99)
        result = Float(iotype='out')
        
        def __init__(self, doc=None):
            super(ParaboloidComponent, self).__init__(doc)
            self.x = numpy.array([10., 10.], dtype=float) # initial guess
            self.result = 0.
            
            self.opt_objective = 0.
            self.opt_design_vars = [2., 3.]
    
        def execute(self):
            """calculate the new objective value"""
            
            self.result = (self.x[0] - 2.0) ** 2 + (self.x[1] - 3.0) ** 2
    
    top = set_as_top(Assembly())
    top.add('comp', ParaboloidComponent())
    top.add('driver', IPOPTdriver())
    top.driver.workflow.add('comp')
    top.driver.print_level = 0

    top.driver.add_objective( 'comp.result' )

    top.driver.add_parameter('comp.x[0]', -100.0, 100.0,
                                      fd_step = .00001)
    top.driver.add_parameter('comp.x[1]', -100.0, 100.0,
                                      fd_step = .00001)
    
    top.driver.add_constraint( 'comp.x[0] - 4.0 > 0.0' )

    top.run()
    
    print "%.4f %.4f %.4f" % ( top.comp.x[0], top.comp.x[1], top.driver.eval_objective() )
    
.. testoutput:: Ipoptdriver
   :hide:
   :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

   4.0000 3.0000 4.0000
