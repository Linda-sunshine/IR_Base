/**
 * 
 */
package structures;

import java.util.Arrays;


/**
 * @author lg5bt
 * basic statistics for each feature
 */
public class _stat {
	int[] m_DF; // document frequency for this feature
	int[] m_ADPDF; // added by Lin, DF got from adaptation data.
	int[] m_TTF; // total term frequency for this feature
	
	public _stat(int classNo){
		m_DF = new int[classNo];
		m_ADPDF = new int[classNo];
		m_TTF = new int[classNo];
	}
	
	public void reset(int classNo) {
		if (m_DF.length<classNo) {
			m_DF = new int[classNo];
			m_ADPDF = new int[classNo];
			m_TTF = new int[classNo];
		} else {
			Arrays.fill(m_DF, 0);
			Arrays.fill(m_ADPDF, 0);
			Arrays.fill(m_TTF, 0);
		}
	}
	
	public int[] getDF(){
		return this.m_DF;
	}
	
	public int[] getADPDF(){
		return this.m_ADPDF;
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
	
	public void addOneADPDF(int index){
		this.m_ADPDF[index]++;
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


