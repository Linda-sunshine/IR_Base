package structures;

import java.util.ArrayList;

import json.JSONException;
import json.JSONObject;
import utils.Utils;

public class AppReviewPost extends Post {
	String m_appId;
	ArrayList<String> content = new ArrayList<String>();
	ArrayList<Integer> contentLabel = new ArrayList<Integer>();
	
	String m_category;
	public String getCategory(){
		return m_category;
	}
	
	public void setCategory(String category){
		m_category = category;
	}
	
	public String getAppId() {
		return m_appId;
	}

	public void setAppId(String appId) {
		this.m_appId = appId;
	}

	String m_appName;
	public String getAppName() {
		return m_appName;
	}

	public void setAppName(String appName) {
		this.m_appName = appName;
	}
	
	String m_versionID;
	public String getVersion() {
		return m_versionID;
	}

	public void setVersionID(String versionID) {
		this.m_versionID = versionID;
	}

	String m_content = "";
	public String getContent(){
		for(String str:content)
			m_content = m_content + " " + str;
		return m_content;
	}
	
	public ArrayList<String> getSentences() {
		if (content==null || content.isEmpty())
			return null;
		return content;
	}
	
	public ArrayList<Integer> getSentenJudgedLabel() {
		if (contentLabel==null || contentLabel.isEmpty())
			return null;
		return contentLabel;
	}

	public void setContent(String content) {
		this.m_content = content;
	}
	
	private int m_rating; 
	public void setRating(int rating) {
		this.m_rating = rating;
	}
	
	public int getRating(){
		return this.m_rating;
	}
	
	public AppReviewPost(String ID) {
		super(ID);
	}

	public AppReviewPost(JSONObject json) {
		super("");
		
		
		try {//special treatment for the overall ratings
			if (json.has("rating")){				
				double label = json.getInt("rating");
				setLabel((int)label); // 1 to 5 actually, but in code 0 to 4
			}
			
			JSONObject sentenceJson = json.getJSONObject("sentences");
			int i = 0;
			while(true){
				String key = "stn_"+i;
				i++;
				JSONObject stnJson = sentenceJson.getJSONObject(key);
				if(stnJson==null)
					break;
				content.add(Utils.getJSONValue(stnJson, "content"));
				int tmp;
				if (stnJson.get("judged") instanceof Integer){
					tmp = (Integer) stnJson.get("judged");
				}else{
					tmp = Integer.parseInt((String) stnJson.get("judged"));
				}
				contentLabel.add(tmp);
				
			}
		
			
		} catch (Exception e) {
			setLabel(1);
			//e.printStackTrace();
		}
		
		
		try {
			JSONObject sentenceJson;
			sentenceJson = json.getJSONObject("sentences");
			String key = "stn_t";
			JSONObject stnJson = sentenceJson.getJSONObject(key);
			content.add(Utils.getJSONValue(stnJson, "content"));
			int tmp;
			if (stnJson.get("judged") instanceof Integer){
				tmp = (Integer) stnJson.get("judged");
			}else{
				tmp = Integer.parseInt((String) stnJson.get("judged"));
			}
			contentLabel.add(tmp);
			
		} catch (JSONException e) {
			//e.printStackTrace();
		}
		
		setAppId(Utils.getJSONValue(json, "appid"));
		setCategory(Utils.getJSONValue(json, "category"));
		setAppName(Utils.getJSONValue(json, "appname"));
		setTitle(Utils.getJSONValue(json, "title"));
		setVersionID(Utils.getJSONValue(json, "version"));
		setRating(Integer.parseInt(Utils.getJSONValue(json, "rating")));
		m_ID = Utils.getJSONValue(json, "reviewid").trim();
	}
	
}
