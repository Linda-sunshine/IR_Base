package structures;

import json.JSONObject;
import utils.Utils;

public class AppReviewPost extends Post {
	String m_appId;
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

	
	
	String m_content;
	public String getContent() {
		if (m_content==null || m_content.isEmpty())
			return null;
		return m_content;
	}

	public void setContent(String content) {
		this.m_content = content;
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
		} catch (Exception e) {
			setLabel(1);
		}
		
		setAppId(Utils.getJSONValue(json, "appid"));
		setAppName(Utils.getJSONValue(json, "appname"));
		setTitle(Utils.getJSONValue(json, "title"));
		setContent(Utils.getJSONValue(json, "content"));
		setVersionID(Utils.getJSONValue(json, "version"));
		
		m_ID = Utils.getJSONValue(json, "reviewid").trim();
	}
	
}
