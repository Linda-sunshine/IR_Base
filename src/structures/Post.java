/**
 * 
 */
package structures;

import json.JSONException;
import json.JSONObject;

/**
 * @author hongning
 * @version 0.1
 * @category data structure
 * data structure for a forum discussion post 
 */
public class Post {
	//unique ID from the corresponding website
	String m_ID;
	public String getID() {
		return m_ID;
	}

	//author's displayed name
	String m_author;
	public String getAuthor() {
		return m_author;
	}
	public void setAuthor(String author) {
		this.m_author = author;
	}

	//unique author ID from the corresponding website
	String m_authorID;	
	public String getAuthorID() {
		return m_authorID;
	}
	public void setAuthorID(String authorID) {
		this.m_authorID = authorID;
	}

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
		if (!content.isEmpty())
			this.m_content = content;
	}

	//timestamp of the post
	String m_date;
	public String getDate() {
		return m_date;
	}
	public void setDate(String date) {
		this.m_date = date;
	}
	
	//post ID that this post is reply to
	String m_replyToID;
	public String getReplyToID() {
		return m_replyToID;
	}
	public void setReplyToID(String replyToID) {
		this.m_replyToID = replyToID;
	}
	
	//only used in eHealth to keep track of reply-to relation
	int m_level; 
	public int getLevel() {
		return m_level;
	}
	public void setLevel(int level) {
		this.m_level = level;
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
	public Post(String ID) {
		m_ID = ID;
	}
	//Constructor.
	public Post(JSONObject json) {
		try {
			if (json.has("Overall")){
				setLabel((int)Double.parseDouble(json.getString("Overall")));
				//System.out.println("label is " + this.m_label);
			}
			if (json.has("Date")){
				setDate(json.getString("Date"));
				//System.out.println("date is " + this.m_date);
			}
			if (json.has("Content")){
				setContent(json.getString("Content"));
				//System.out.println("content is " + this.m_content);
			}
			
//			//Not used currently.
//			if (json.has(m_ID)){
//				m_ID = json.getString("postID");
//				System.out.println("postID is " + this.m_ID);
//			}
//			if (json.has("Author")){
//				setAuthor(json.getString("Author"));
//				System.out.println("Author is " + this.m_author);
//			}
//			if (json.has("AuthorID")){
//				setAuthorID(json.getString("AuthorID"));
//				System.out.println("AuthorID is " + this.m_authorID);
//			}
//			if (json.has("replyTo"))
//				setReplyToID(json.getString("replyTo"));
//			
//			if (json.has("Title")){
//				setTitle(json.getString("Title"));
//			}
		} catch (JSONException e) {
			//e.printStackTrace();
		}
	}
	
	public JSONObject getJSON() throws JSONException {
		JSONObject json = new JSONObject();
		json.put("postID", m_ID);//must contain
		json.put("author", m_author);//must contain
		json.put("authorID", m_authorID);//must contain
		json.put("replyTo", m_replyToID);//might be missing
		json.put("date", m_date);//must contain
		json.put("title", m_title);//might be missing
		json.put("content", m_content);//must contain
		
		return json;
	}
}
