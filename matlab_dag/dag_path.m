classdef dag_path
  %DAG_PATH Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods(Static)
    function add ()
      vl_setupnn();
      rp = dag_path.root();
      addpath(fullfile(rp, 'matlab_dag')) ;
    end
    
    function rp = root ()
      rp = fileparts(fileparts(mfilename('fullpath'))) ;
    end
  end
  
end

