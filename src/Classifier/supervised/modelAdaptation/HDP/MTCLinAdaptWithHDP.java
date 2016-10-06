package Classifier.supervised.modelAdaptation.HDP;

import java.util.HashMap;
import cern.jet.random.tfloat.FloatUniform;
import structures._HDPThetaStar;
import structures._Review;
import structures._thetaStar;
import structures._Review.rType;
import structures._SparseFeature;
import utils.Utils;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.DirichletProcess.CLinAdaptWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess.MTCLinAdaptWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess._DPAdaptStruct;
/***
 * This class implements the MTCLinAdapt with HDP added.
 * Currently, each review is assigned to one group and each user is a mixture of the components.
 * @author lin
 *
 */
public class MTCLinAdaptWithHDP extends MTCLinAdaptWithDP {
	
	public MTCLinAdaptWithHDP(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel,
			String featureGroupMap, String featureGroup4Sup) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap,
				featureGroup4Sup);
	}
	
	
}
