/**
 * 
 */
package structures;

/**
 * @author lingong
 * Feature structure for sparse feature representation
 */
public class _SparseFeature implements Comparable<_SparseFeature> {
	private String content; //Content of the feature.
	private int m_index; // Index of the feature
	private double m_value; // Value of the feature (non-zero)
	private double m_norm_value; //Normalized value of the feature.
	
	//Constructor.
	public _SparseFeature(){
		this.content = "";
		this.m_index = 0;//why I commented this line???
		this.m_value = 0;
		this.m_norm_value = 0;
	}
	
	//Constructor.
	public _SparseFeature(int index, String content) {
		this.content = content;
		this.m_index = index;// why I commented this line???
		this.m_value = 0;
		this.m_norm_value = 0;
	}
	
	//Get the content of the feature.
	public String getContent(){
		return this.content;
	}
	
	public String setContent(String content){
		this.content = content;
		return this.content;
	}
	
	//Get the index of the feature.
	public int getIndex(){
		return this.m_index;
	}
	
	//Set the index for the feature.
	public int setIndex(int index){
		this.m_index = index;
		return this.m_index;
	}
	
	//Get the value of the feature.
	public double getValue(){
		return this.m_value;
	}
	
	//Set the value for the feature.
	public double setValue(double value){
		this.m_value = value;
		return this.m_value;
	}	
	
	//Get the normalized value for the feature.
	public double getNormValue(){
		return this.m_norm_value;
	}
	
	//Set the normalized value for the feature.
	public double setNormValue(double normValue){
		this.m_norm_value = normValue;
		return this.m_norm_value;
	}

	@Override
	public int compareTo(_SparseFeature sfv) {
		return m_index - sfv.m_index;
	}
	
	
}