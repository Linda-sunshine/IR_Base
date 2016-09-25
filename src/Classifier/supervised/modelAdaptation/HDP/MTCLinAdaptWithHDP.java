package Classifier.supervised.modelAdaptation.HDP;

import java.util.HashMap;

import Classifier.supervised.modelAdaptation.DirichletProcess.MTCLinAdaptWithDP;

public class MTCLinAdaptWithHDP extends MTCLinAdaptWithDP {

	public MTCLinAdaptWithHDP(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel,
			String featureGroupMap, String featureGroup4Sup) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap,
				featureGroup4Sup);
		// TODO Auto-generated constructor stub
	}

}
