/**
 * 
 */
package structures;

import java.util.Arrays;
import java.util.HashMap;


/**
 * @author lg5bt
 * basic statistics for each feature
 */
public class _stat {
	double[] m_gloveVec;//word embedding

	int[] m_DF; // document frequency for this feature
	int[] m_TTF; // total term frequency for this feature

//	public HashMap<String, Double> m_wordSimMap;
	
	public _stat(int classNo){
		m_DF = new int[classNo];
		m_TTF = new int[classNo];
	}
	
	public void reset(int classNo) {
		if (m_DF.length<classNo) {
			m_DF = new int[classNo];
			m_TTF = new int[classNo];
		} else {
			Arrays.fill(m_DF, 0);
			Arrays.fill(m_TTF, 0);
		}
	}

	public void setSimMap(){
//		m_wordSimMap = new HashMap<String, Double>();
	}

	public void setM_gloveVec(double[] gloveVec){
		int gloveLen = gloveVec.length;
		m_gloveVec = new double[gloveLen];
		for(int i=0; i<gloveLen; i++){
			m_gloveVec[i] = gloveVec[i];
		}
	}

	public double[] getM_gloveVec(){
		return m_gloveVec;
	}

	public int[] getDF(){
		return this.m_DF;
	}
	
	public void setDF(int[] DFs) {
		if (DFs.length != m_DF.length)
			return;
		System.arraycopy(DFs, 0, m_DF, 0, DFs.length);
	}
	
	public int[] getTTF(){
		return this.m_TTF;
	}
	
	//The DF(The document frequency) of a feature is added by one.
	public void addOneDF(int index){
		this.m_DF[index]++;
	}
	
	//The TTF(Total term frequency) of a feature is added by one.
	public void addOneTTF(int index){
		this.m_TTF[index]++;
	}
	
	public void minusOneDF(int index){
		this.m_DF[index]--;
	}

	public void minusNTTF(int index, double n){
		this.m_TTF[index] -= n;
	}
}


