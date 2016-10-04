package Classifier.supervised.modelAdaptation.HDP;

import java.util.HashMap;

import structures._HDPThetaStar;
import structures._User;
import structures._thetaStar;
import Classifier.supervised.modelAdaptation.DirichletProcess._DPAdaptStruct;

public class _HDPAdaptStruct extends _DPAdaptStruct {
	
	_HDPThetaStar  m_hdpThetaStar = null;
	
	/**key: global component parameter \phi; 
	 * val: the number of local groups belonging to the global component.**/
	protected HashMap<_HDPThetaStar, Integer> m_hMap;
	
	/**key: global component parameter \phi; 
	 * val: member size.**/
	protected HashMap<_HDPThetaStar, Integer> m_hdpThetaMemSizeMap;
	
	public _HDPAdaptStruct(_User user) {
		super(user);
		m_hMap = new HashMap<_HDPThetaStar, Integer>(); 
		m_hdpThetaMemSizeMap = new HashMap<_HDPThetaStar, Integer>();
	}

	//Return the number of members in the given thetaStar.
	public int getHDPThetaMemSize(_HDPThetaStar s){
		if(m_hdpThetaMemSizeMap.containsKey(s))
			return m_hdpThetaMemSizeMap.get(s);
		else 
			return 0;
	}
	//Add a new component to the 
	public void addHDPThetaStar(_HDPThetaStar s){
		m_hdpThetaMemSizeMap.put(s, 1);
	}
	public void incHDPThetaStarMemSize(_HDPThetaStar s){
		int val = m_hdpThetaMemSizeMap.get(s)+1;
		m_hdpThetaMemSizeMap.put(s, val);
	}
	
	public HashMap<_HDPThetaStar, Integer> getHMap(){
		return m_hMap;
	}
	
	public HashMap<_HDPThetaStar, Integer> getHDPThetaMemSizeMap(){
		return m_hdpThetaMemSizeMap;
	}
	
	public _HDPThetaStar getThetaStar(){
		return m_hdpThetaStar;
	}
	
}
