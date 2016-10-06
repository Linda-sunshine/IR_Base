package mains;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import Classifier.supervised.SVM;
import Classifier.supervised.liblinear.Feature;
import Classifier.supervised.liblinear.FeatureNode;
import Classifier.supervised.liblinear.Linear;
import Classifier.supervised.liblinear.Model;
import Classifier.supervised.liblinear.Parameter;
import Classifier.supervised.liblinear.Problem;
import Classifier.supervised.liblinear.SolverType;

/***
 * In the main, we want to see if SVM supports one class classificatio.
 * @author lin
 *
 */

public class SVMTestMain {	
	Feature[][] m_fvs;
	double[] m_ys;
	int count = 0;
	
	public SVMTestMain(int num){
		m_fvs = new Feature[num][];
		m_ys = new double[num];
	}
	public void loadFvsYs(String filename){
		if(filename==null || filename.isEmpty())
			return;
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			while((line = reader.readLine()) != null){
				String[] tmp = line.split(",");
				m_ys[count] = Integer.valueOf(tmp[0]);
				m_fvs[count] = analyzeFvs(tmp);
				count++;
			}
			System.out.format("%d instances are loaded.\n", count);
			reader.close();
		}
		catch (IOException e){
			System.err.format("Failed to open the file %s!", filename);
			return;
		}
	}
	//Translate the input string array as an instance.
	public Feature[] analyzeFvs(String[] tmp){
		if(tmp.length %2 != 1 || tmp.length <1)
			return null;
		int dimension = (tmp.length - 1)/2;
		Feature[] res = new Feature[dimension];
		for(int i=0; i<dimension; i++){
			FeatureNode node = new FeatureNode(Integer.valueOf(tmp[2*i+1]), Double.valueOf(tmp[2*(i+1)]));
			res[i] = node;
		}
		return res;
	}
	/***
	 * @param bias
	 * @param method==1, <-x, -y>; method==2, part<x, y>+part<-x, -y>; else<x, y>.
	 * @return
	 */
	public void train(double bias, int method){
		SolverType type = SolverType.L2R_L1LOSS_SVC_DUAL;
		Problem libProblem = new Problem();
		libProblem.l = m_fvs.length;
		libProblem.n = 11;
		libProblem.bias = bias;
		Feature[][] x = new Feature[m_fvs.length][];
		double[] y = new double[m_fvs.length];
		if(method == 1){
			for(int i=0; i<m_fvs.length; i++){
				x[i] = reverse(m_fvs[i]);
				y[i] = -m_ys[i];
			}
		} else if(method == 2){
			Random r = new Random();
			for(int i=0; i<m_fvs.length; i++){
				if(r.nextDouble()>0.5){
					x[i] = reverse(m_fvs[i]);
					y[i] = -m_ys[i];
				}
				else{
					x[i] = m_fvs[i];
					y[i] = m_ys[i];
				}
			}
		} else{
			x = m_fvs;
			y = m_ys;
		}
		
		libProblem.x = x;
		libProblem.y = y;
		Model model = Linear.train(libProblem, new Parameter(type, 1, SVM.EPS));
		double[] weights = model.getWeights();
		for(double w: weights)
			System.out.print(w+"\t");
		System.out.println();
		System.out.println("bias is "+model.getBias());
//		return weights;
	}
	
	public void trainDouble(){
		SolverType type = SolverType.L2R_L1LOSS_SVC_DUAL;
		Problem libProblem = new Problem();
		libProblem.l = m_fvs.length*2;
		libProblem.n = 11;
		libProblem.bias = -1;
		Feature[][] x = new Feature[m_fvs.length*2][];
		double[] y = new double[m_fvs.length*2];
		for(int i=0; i<m_fvs.length; i++){
			x[i] = m_fvs[i];
			y[i] = m_ys[i];
			x[i+m_fvs.length] = reverse(m_fvs[i]);
			y[i+m_fvs.length] = -m_ys[i];
		}
		
		libProblem.x = x;
		libProblem.y = y;
		Model model = Linear.train(libProblem, new Parameter(type, 0.5, SVM.EPS));
		double[] weights = model.getWeights();
		for(double w: weights)
			System.out.print(w+"\t");
		System.out.println();
		System.out.println("Bias is "+model.getBias());
//		return weights;
	}
	public Feature[] reverse(Feature[] fv){
		Feature[] res = new Feature[fv.length];
		for(int i=0; i<fv.length; i++){
			res[i] = new FeatureNode(fv[i].getIndex(), -fv[i].getValue());
		}
		return res;
	}
	public static void main(String[] args){
		SVMTestMain test = new SVMTestMain(1000);
		
		test.loadFvsYs("./data/RankSVMDataFile1027.csv");
//		test.trainDouble();
//		test.trainDouble();
//		System.out.println("<x, y> is applied in this method.");
//		test.train(-1, 0);
//		System.out.println("<-x, -y> is applied in this method.");
		test.train(-1, 1);
		System.out.println("part<x, y> + part<-x, -y> is applied in this method.");
//		test.train(-1, 2);
		
//		HashMap<String, Integer> test1 = new HashMap<String, Integer>();
//		test1.put("a", 2);
//		test1.put("b", 3);
//		test1.put("c", 4);
//		Iterator<Entry<String, Integer>> it = test1.entrySet().iterator();
//		while(it.hasNext()){
//			Map.Entry<String, Integer> pair = (Entry<String, Integer>) it.next();
//			System.out.println(pair.getKey()+","+pair.getValue());
//		}
		
	}
}
