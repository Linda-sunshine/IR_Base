package Classifier.supervised.modelAdaptation.HDP;

import structures._User;
import Classifier.supervised.modelAdaptation.DirichletProcess._DPAdaptStruct;

public class _HDPAdaptStruct extends _DPAdaptStruct {
	int m_h;// the number of local groups for each user.
	
	public _HDPAdaptStruct(_User user) {
		super(user);
	}

}
