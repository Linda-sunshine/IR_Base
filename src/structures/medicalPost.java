/**
 * 
 */
package structures;

import json.JSONException;
import json.JSONObject;
import utils.Utils;

/**
 * @author Md Mustafizur Rahman
 * @version 0.1
 * @category data structure
 * data structure for a forum discussion post 
 */
public class medicalPost {
		//post title (might not be available in some medical forums)
	String m_title;//not available in WebMD
	public String getTitle() {
		return m_title;
	}
	public void setTitle(String title) {
		if (!title.isEmpty())
			this.m_title = title;
	}

	//post content
	String m_content;
	public String getContent() {
		return m_content;
	}
	public void setContent(String content) {
		if (!content.isEmpty()) {
			this.m_content = Utils.cleanHTML(content);
		}
	}

	//Used for classification.
	int m_label;
	public void setLabel(int overall){
		this.m_label = overall;
	}
	public int getLabel(){
		return this.m_label;
	}
	
	//Constructor.
	public medicalPost(JSONObject json) {
		try {//special treatment for the overall ratings
			setContent(Utils.getJSONValue(json, "content"));
			setTitle(Utils.getJSONValue(json, "title"));
		} 
		catch (Exception e) {
			System.err.println("Error Reading JSON Object!!");
			e.printStackTrace();
		}
		
		
	}
	
	public JSONObject getJSON() throws JSONException {
		JSONObject json = new JSONObject();
		json.put("title", m_title);//might be missing
		json.put("content", m_content);//must contain
		json.put("type", m_label);//must contain
		return json;
	}
}
