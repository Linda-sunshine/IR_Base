/**
 * 
 */
package structures;


/**
 * @author lg5bt
 * basic statistics for each feature
 */
public class _stat {
	int[] m_DF; // document frequency for this feature
	int[] m_TTF; // total term frequency for this feature
	//Since we do not have a general method for judging classes, leave it here.
	int[][] m_counts; // frequency with respect to class label

	public _stat(int classNo){
		m_DF = new int[classNo];
		m_TTF = new int[classNo];
	}
	
	public int[] getDF(){
		return this.m_DF;
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
	
	public void initCount(int classNo){
		this.m_counts = new int[2][classNo];
	}
	
	//Set the DF count of each feature.
	public void setCounts(int[] classMemberNo){
		for(int i = 0; i < classMemberNo.length; i++){
			this.m_counts[0][i] = this.m_DF[i];
			this.m_counts[1][i] = classMemberNo[i]- this.m_DF[i];
		}
	}
	
	public int[][] getCounts(){
		return this.m_counts;
	}
}


