package structures;

import java.util.Arrays;
import java.util.HashMap;

import Analyzer.ParentChildAnalyzer;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.jet.random.tdouble.Normal;

public class _ChildDoc4ProbitModel extends _ChildDoc{
	double[][] m_probitFvcts; // to facilitate computation
	public HashMap<Integer, double[]> m_fixedMuPartMap;
	public HashMap<Integer, Double> m_fixedSigmaPartMap;
	
	public _ChildDoc4ProbitModel(int ID, String name, String title, String source, int ylabel){
		super(ID, name, title, source, ylabel);
		
		m_fixedMuPartMap = new HashMap<Integer, double[]>();//from word index to a vector
		m_fixedSigmaPartMap = new HashMap<Integer, Double>();
	}
	
	void setFixedFeatureValueMap(){
		DenseDoubleMatrix2D AMtx = new DenseDoubleMatrix2D(ParentChildAnalyzer.ChildDocFeatureSize, ParentChildAnalyzer.ChildDocFeatureSize);
		DoubleMatrix2D AMtxInv;
		DenseDoubleAlgebra mtxOpt = new DenseDoubleAlgebra();
		double[] tmpVct = new double[ParentChildAnalyzer.ChildDocFeatureSize], rltVct; 
		boolean[] bitMap = new boolean[getDocLength()];
		_Word word;
		int wid, wIndex;
		double val;
		
		for(int n=0; n<m_words.length; n++){
			word = m_words[n];
			wid = word.getIndex();
			
			if (m_fixedMuPartMap.containsKey(wid))
				continue;			
			
			// step1: compute A matrix and its inverse
			calcAMatrix(AMtx, n);
			AMtxInv = mtxOpt.inverse(AMtx);
			
			// step2: tmpVct= f^T \times A^{-1}
			for(int k=0; k<ParentChildAnalyzer.ChildDocFeatureSize; k++) {
				val = 0;
				for(int l=0; l<ParentChildAnalyzer.ChildDocFeatureSize; l++)
					val += m_probitFvcts[n][l] * AMtxInv.getQuick(l, k);
				tmpVct[k] = val;
			}
			
			// step3: tmpVct \times F_{-n}^T
			rltVct = new double[getDocLength()];//length of sparse vector
			Arrays.fill(bitMap, false);
			for(int m=0; m<m_words.length; m++) {
				wIndex = m_words[m].getLocalIndex();
				if (m==n || bitMap[wIndex])//will leave the corresponding dimension zero
					continue;
				
				val = 0;
				for(int l=0; l<ParentChildAnalyzer.ChildDocFeatureSize; l++)
					val += tmpVct[l] * m_probitFvcts[m][l];				
				
				rltVct[wIndex] = val;
				bitMap[wIndex] = true;
			}
			
			// step 4: tmpVct \times f^T
			val = 0;
			for(int k=0; k<ParentChildAnalyzer.ChildDocFeatureSize; k++) 
				val += m_probitFvcts[n][k] * tmpVct[k];
				
			m_fixedMuPartMap.put(wid, rltVct);
			m_fixedSigmaPartMap.put(wid, Math.sqrt(val));
		}
	}
	
	void calcAMatrix(DenseDoubleMatrix2D AMtx, int wid) {
		for(int i=0; i<ParentChildAnalyzer.ChildDocFeatureSize; i++) {
			for(int j=0; j<ParentChildAnalyzer.ChildDocFeatureSize; j++) {				
				double sum = 0;
				if (i==j)
					sum = 1; //the covariance of \lambda's prior is fixed to an identity matrix 
				
				for(int l=0; l<m_words.length; l++) {
					if (l == wid)
						continue;//skip the corresponding word
					sum += m_probitFvcts[l][i] * m_probitFvcts[l][j];
				}
				AMtx.setQuick(i, j, sum);
			}	
		}
		
	}
	
	@Override
	void createSpace(int k, double alpha) {
		super.createSpace(k, alpha);
		
		m_probitFvcts = new double[getTotalDocLength()][];
	}
	
	@Override
	public void setTopics4Gibbs(int k, double alpha){
		createSpace(k, alpha);
		
		int wid, tid, xid, wIndex = 0, localIndex = 0;
		double xVal;
		for(_SparseFeature fv: m_x_sparse){
			wid = fv.getIndex();
			
			for(int j=0; j<fv.getValue(); j++){
				// U(0,k-1) uniform initialization 
				tid = m_rand.nextInt(k);
				// N(0,1) gaussian initialization
				xVal = Normal.staticNextDouble(0, 1);
				m_words[wIndex] = new _Word(wid, tid, xVal, localIndex, fv.getValues());
				
				xid = m_words[wIndex].getX();
				m_xTopicSstat[xid][tid] ++;
				m_xSstat[xid] ++;
				m_probitFvcts[wid] = fv.getValues();
				
				wIndex ++;
			}
			localIndex ++;
		}
		
		setFixedFeatureValueMap();
	}
}
