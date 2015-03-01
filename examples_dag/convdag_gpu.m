classdef convdag_gpu < convdag
  %CONVDAG_GPU Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods % wait(gpuDevice) when fprop and bprop
    function ob = do_fprop_bprop(ob)
      % fprop
      for i = 1 : numel(ob.tfs)
        ob.tfs{i} = fprop( ob.tfs{i} );
        wait(gpuDevice);
      end % for i
      
      % bprop
      for i = numel(ob.tfs) : -1 : 1
        ob.tfs{i} = bprop( ob.tfs{i} );
        wait(gpuDevice);
      end
    end % do_fprop_bprop
    
    function ob = do_fprop_bprop_tight_mem(ob)
      % fprop
      for i = 1 : numel(ob.tfs)
        ob.tfs{i} = fprop( ob.tfs{i} );
        wait(gpuDevice);
        ob.tfs{i} = cl_i_a( ob.tfs{i} );
      end % for i
      
      % bprop
      for i = numel(ob.tfs) : -1 : 1
        ob.tfs{i} = bprop( ob.tfs{i} );
        wait(gpuDevice);
        ob.tfs{i} = cl_o_d( ob.tfs{i} );
      end
    end % do_fprop_bprop_tight_mem
  end
  
  methods % need be overrided in derived class
    function ob = init_dag (ob)
      error('Call this in a derived class :)');
    end
    
    function ob = set_node_src (ob, X_bat, Y_bat)
      error('Call this in a derived class :)');
    end
    
    function ob = set_node_sink (ob, varargin)
      error('Call this in a derived class :)');
    end

  end % methods
  
end

