.. _texnet:

=========================
TexNet SeisComP addons documentation
=========================

This is the documentation for the SeisComP modules, tools, packages and plugins developed by TexNet.

.. _fig-texnet:

.. figure:: apps/media/Texnet-logo.png
   :width: 60%
   :align: center

About 
-----

This documentation provides a comprehensive reference for the modules 
and tools developed by the Texas Seismological Network (TexNet) 
at the Bureau of Economic Geology, The University of Texas at Austin. 
All components are designed to integrate seamlessly with `SeisComP <https://www.seiscomp.de/>`_ 
(`official documentation <https://docs.gempa.de/seiscomp/current/>`_), enhancing its 
capabilities for real-time seismic monitoring, data quality control, and research applications.

The modules documented here include implementations of advanced algorithms 
for seismic phase picking, event detection, quality control, and moment tensor inversion, 
among other functionalities. Each application is accompanied by a public Git repository, 
ensuring transparency, reproducibility, and community access. While developed to address 
specific operational and research needs of TexNet, these tools are flexible and can be 
adapted to broader seismic network environments.

This site serves as a central reference for installation, configuration, and usage of these tools. 
In addition, it provides theoretical background on the methods implemented, giving users both the 
practical guidelines and the conceptual understanding needed to fully leverage TexNet’s developments 
within SeisComP-based infrastructures.



Table of Contents
-----------------

.. toctree::
   :maxdepth: 1
   :caption: Modules

   /apps/scmlpick
   .. /apps/sceewv
   .. /apps/scqcalert

.. toctree::
   :maxdepth: 1
   :caption: Repositories

   scmlpick git <https://github.com/ut-beg-texnet/scmlpick.git>

.. toctree::
   :maxdepth: 1
   :caption: Plugins