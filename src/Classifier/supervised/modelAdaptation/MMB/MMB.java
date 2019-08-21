package Classifier.supervised.modelAdaptation.MMB;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import structures._HDPThetaStar;
import structures._User;

import java.util.ArrayList;
import java.util.HashMap;

public class MMB extends CLRWithMMB {

    public MMB(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel,
                      double[] betas) {
        super(classNo, featureSize, featureMap, globalModel, betas);
    }

    @Override
    public void loadUsers(ArrayList<_User> userList) {
        m_userList = new ArrayList<_AdaptStruct>();
        // Init each user.
        for(_User user:userList){
            m_userList.add(new _MMBAdaptStruct(user, m_dim));
        }
        m_pWeights = new double[m_gWeights.length];
        m_indicator = new _HDPThetaStar[m_userList.size()][m_userList.size()];
    }
}
