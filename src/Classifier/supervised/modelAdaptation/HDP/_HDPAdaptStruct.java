package Classifier.supervised.modelAdaptation.HDP;

import java.util.HashMap;

import structures._User;
import structures._thetaStar;
import Classifier.supervised.modelAdaptation.DirichletProcess._DPAdaptStruct;

public class _HDPAdaptStruct extends _DPAdaptStruct {
	//key: global component parameter \phi; val: the number of local groups belonging to the global component.
	protected HashMap<_thetaStar, Integer> m_hMap;
	//key: global component parameter \phi; val: member size.
	protected HashMap<_thetaStar, Integer> m_thetaStarMemSizeMap;
	
	public _HDPAdaptStruct(_User user) {
		super(user);
		m_hMap = new HashMap<_thetaStar, Integer>(); 
		m_thetaStarMemSizeMap = new HashMap<_thetaStar, Integer>();
	}

	//Return the number of members in the given thetaStar.
	public int getThetaStarMemSize(_thetaStar s){
		if(m_thetaStarMemSizeMap.containsKey(s))
			return m_thetaStarMemSizeMap.get(s);
		else 
			return 0;
	}
	//Add a new component to the 
	public void addThetaStar(_thetaStar s){
		m_thetaStarMemSizeMap.put(s, 1);
	}
	public void incThetaStarMemSize(_thetaStar s){
		int val = m_thetaStarMemSizeMap.get(s)+1;
		m_thetaStarMemSizeMap.put(s, val);
	}
	
	public HashMap<_thetaStar, Integer> getHMap(){
		return m_hMap;
	}
}
