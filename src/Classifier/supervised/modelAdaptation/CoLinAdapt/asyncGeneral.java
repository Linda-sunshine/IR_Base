package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.HashMap;

import structures._PerformanceStat;
import structures._Review;
import structures._PerformanceStat.TestMode;
import Classifier.supervised.modelAdaptation.RegLR.asyncRegLR;

public class asyncGeneral extends asyncLinAdapt {
	String m_dataset;
	String m_method;

	public asyncGeneral(String method, int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel,
			String featureGroupMap) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap);
		m_method = method;
	}
	
	@Override
	public String toString() {
		return String.format("async%s[dim:%d,eta1:%.3f,eta2:%.3f]", m_method, m_dim, m_eta1, m_eta2);
	}
	
	public void setDataset(String data){
		m_dataset = data;
	}
	
	//this is online training in each individual user
	@Override
	public double train(){
		int predL, trueL;
		_Review doc;
		double val = 0;
		_PerformanceStat perfStat;
		_LinAdaptStruct user;

		initLBFGS();
		init();
		try {
			m_writer = new PrintWriter(new File(String.format("%s_online_%s.txt", m_dataset, m_method)));
			for(int i=0; i<m_userList.size(); i++) {
				user = (_LinAdaptStruct)m_userList.get(i);
				while(user.hasNextAdaptationIns()) {
					// test the latest model before model adaptation
					if (m_testmode != TestMode.TM_batch &&(doc = user.getLatestTestIns()) != null) {
						perfStat = user.getPerfStat();
						val = logit(doc.getSparse(), user);
						predL = val > 0.5 ? 1 : 0;
						trueL = doc.getYLabel();
						perfStat.addOnePredResult(predL, trueL);
						m_writer.format("%s\t%d\t%.4f\t%d\t%d\n", user.getUserID(), doc.getID(), val, predL, trueL);
					} // in batch mode we will not accumulate the performance during adaptation		
					user.nextAdaptationIns();
				}
			}
			m_writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
			
		setPersonalizedModel();
		return 0;//we do not evaluate function value
	}	
	
	public void loadGlobal(String filename){
		if (filename==null)
			return;	
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			int pos = 0;
			m_gWeights = new double[m_featureSize+1];//to include the bias term
			while((line=reader.readLine()) != null) {
				m_gWeights[pos++] = Double.valueOf(line);
			}
			reader.close();
		} catch(IOException e){
			System.err.format("[Error]Fail to open file %s.\n", filename);
		}
	}
}
