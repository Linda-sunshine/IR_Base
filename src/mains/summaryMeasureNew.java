/**
 * 
 */
package mains;

import java.awt.List;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.lang.reflect.Array;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.LinkedList;
import java.util.Map;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map.Entry;
import java.util.Random;
import java.util.concurrent.locks.ReentrantReadWriteLock.ReadLock;
import java.util.Set;

public class summaryMeasureNew {
	
	
	private HashMap<String,String> HTSM_summary = new HashMap<String, String>();
	private HashMap<String,String> HTMM_summary = new HashMap<String, String>();
	private HashMap<String,String> ASUM_summary = new HashMap<String, String>();
	
	private HashMap<String, ArrayList<String>> list =new HashMap<String, ArrayList<String>>();
	
	
	private HashMap<String, ArrayList<String>> HTSM =new HashMap<String, ArrayList<String>>();
	private HashMap<String, ArrayList<String>> HTMM =new HashMap<String, ArrayList<String>>();
	private HashMap<String, ArrayList<String>> ASUM =new HashMap<String, ArrayList<String>>();
	
	private ArrayList<String> modelSequence = new ArrayList<String>();

	private int number_of_topics= 0;
	private int number_of_Total_sentences = 0;
	private String category;
	
	private double HTSM_score = 0;
	private double HTMM_score = 0;
	private double ASUM_score = 0;
	


	private String cameraTopic[]= {"battery", "screen","viewfinder","record","mode","shutter","memori","zoom","len","picture","price","Instruction","video","battery", "screen","viewfinder","record","mode","shutter","memori","zoom","len","picture","price","Instruction","video"};
	private String phoneTopic[]= {"screen","app","price","battery","sound","camera","storage","call","servic","cpu","keyboard","design","text","screen","app","price","battery","sound","camera","storage","call","servic","cpu","keyboard","design","text"};
	private String tabletTopic[]= {"screen","app","price","battery","sound","camera","cpu","microsd","internet","keyboard","bluetooth","usb","gps","servic","mous","screen","app","price","battery","sound","camera","cpu","microsd","internet","keyboard","bluetooth","usb","gps","servic","mous"};
	private String tvTopic[]= {"screen","price","sound","servic","picture","quality","app","connection","screen","price","sound","servic","picture","quality","app","connection"};

	
	private Random r;
	
	String tabletProductList[] = {"B008DWG5HE","B00CYQPM42","B007P4YAPK"};
	String cameraProductList[] = {"B005IHAIMA","B002IPHIEG","B00DMS0LCO"};
	String phoneProductList[] = {"B00COYOAYW","B004T36GCU","B008HTJLF6"};
	String tvProductList[] = {"B0074FGLUM","B00BCGROJG","B00AOA9BL0"};
	
	HashMap<String,String> tabletProductNames = new HashMap<String,String>();
	HashMap<String,String> phoneProductNames = new HashMap<String,String>();
	HashMap<String,String> cameraProductNames = new HashMap<String,String>();
	HashMap<String,String> tvProductNames = new HashMap<String,String>();
	
	
	HashMap<String,String> tabletProductKeys = new HashMap<String,String>();
	HashMap<String,String> phoneProductKeys = new HashMap<String,String>();
	HashMap<String,String> cameraProductKeys = new HashMap<String,String>();
	HashMap<String,String> tvProductKeys = new HashMap<String,String>();
	
	HashMap<String,String> ProductKeys = new HashMap<String,String>();
	
	//	{"Samsung Galaxy Note 10.1","Amazon Kindle Fire HDX","ASUS Transformer Tablet"};
	
	
	
	private PrintWriter resultWriter;
	private PrintWriter surveyWriter;

	public void setCategory (String category){
		this.category = category;
		this.number_of_Total_sentences = 0;
		
		if(category.equalsIgnoreCase("camera")){
			ProductKeys = cameraProductKeys;
		}
		else if(category.equalsIgnoreCase("phone")){
			ProductKeys = phoneProductKeys;
		}
		else if(category.equalsIgnoreCase("tablet")){
			ProductKeys = tabletProductKeys;
		}
		else if(category.equalsIgnoreCase("tv")){
			ProductKeys = tvProductKeys;
		}
		
		
	}
	
	public summaryMeasureNew(){
		r = new Random();
		
		tabletProductNames.put("B008DWG5HE", "Samsung Galaxy Note 10.1");
		tabletProductNames.put("B00CYQPM42", "Amazon Kindle Fire HDX");
		tabletProductNames.put("B007P4YAPK", "ASUS Transformer Tablet");
		
		phoneProductNames.put("B00COYOAYW", "Nokia Lumia 521");
		phoneProductNames.put("B004T36GCU", "HTC A9192 Phone");
		phoneProductNames.put("B008HTJLF6", "Samsung Galaxy S III");
		
		cameraProductNames.put("B005IHAIMA", "Sony NEX-5N 16.1 MP Compact Interchangeable Lens Camera");
		cameraProductNames.put("B002IPHIEG", "Sony Cybershot DSC");
		cameraProductNames.put("B00DMS0LCO", "Canon EOS 70D Digital SLR Camera");
		
		tvProductNames.put("B0074FGLUM", "Samsung UN50EH5300 50-Inch 1080p 60Hz LED HDTV");
		tvProductNames.put("B00BCGROJG", "Samsung Ultra Slim Smart LED HDTV 2013 Model");
		tvProductNames.put("B00AOA9BL0", "XFINITY TV Go");
		
		
		tabletProductKeys.put("Samsung Galaxy Note 10.1","B008DWG5HE");
		tabletProductKeys.put("Amazon Kindle Fire HDX","B00CYQPM42");
		tabletProductKeys.put("ASUS Transformer Tablet","B007P4YAPK");
		
		phoneProductKeys.put("Nokia Lumia 521","B00COYOAYW");
		phoneProductKeys.put( "HTC A9192 Phone","B004T36GCU");
		phoneProductKeys.put( "Samsung Galaxy S III","B008HTJLF6");
		
		cameraProductKeys.put("Sony NEX-5N 16.1 MP Compact Interchangeable Lens Camera","B005IHAIMA");
		cameraProductKeys.put("Sony Cybershot DSC","B002IPHIEG");
		cameraProductKeys.put("Canon EOS 70D Digital SLR Camera","B00DMS0LCO");
		
		tvProductKeys.put("Samsung UN50EH5300 50-Inch 1080p 60Hz LED HDTV","B0074FGLUM");
		tvProductKeys.put("Samsung Ultra Slim Smart LED HDTV 2013 Model","B00BCGROJG");
		tvProductKeys.put("XFINITY TV Go","B00AOA9BL0");
	}
	
	public void setResultWriter(String path){
		try{
			resultWriter = new PrintWriter(new File(path));
			
		}
		catch(Exception e){
			System.err.println(path+" Not found!!");
		}
	}
	
	
public void writeCSV(String path, String category, int[] topicID){
		
		PrintWriter writer = null;
		String [] topics = null;
		HashMap<String,String> productNames = new HashMap<String,String>();
		try{
			writer = new PrintWriter(new File(path));
		}
		catch(Exception e){
			System.err.println(path+" Not found!!");
			e.printStackTrace();
			
		}
			if(category.equalsIgnoreCase("camera"))
				topics = cameraTopic;
			else if(category.equalsIgnoreCase("phone"))
				topics = phoneTopic;
			else if(category.equalsIgnoreCase("tablet"))
				topics = tabletTopic;
			else if(category.equalsIgnoreCase("tv"))
				topics = tvTopic;
			
			
			String productList [] = null;
			if(category.equalsIgnoreCase("camera")){
				topics = cameraTopic;
				productList = cameraProductList;
				productNames = cameraProductNames;
			}
			else if(category.equalsIgnoreCase("phone")){
				topics = phoneTopic;
				productList = phoneProductList;
				productNames = phoneProductNames;
			}
			else if(category.equalsIgnoreCase("tablet")){
				topics = tabletTopic;
				productList = tabletProductList;
				productNames = tabletProductNames;
			}
			else if(category.equalsIgnoreCase("tv")){
				topics = tvTopic;
				productList = tvProductList;
				productNames = tvProductNames;
			}
			
			//writer.write("Under each cate");
			for(String prodID:productList){
				
				
				for(int topic = 0; topic<topicID.length; topic++){
					int posTopic = topicID[topic];
					writer.write("Product name:"+productNames.get(prodID)+"\n");
					writer.write("Topic ID:"+ posTopic+"\n");
					writer.write("Tell me something POSITIVE about "+ topics[posTopic].toUpperCase()+" Aspect \n");
					String key = prodID+posTopic;
					ArrayList<String> sentences = list.get(key);
					int sentenceCounter = 0;
					if(sentences==null)
						System.out.println("NULL");
					for(String sentence:sentences){
						System.out.println("sen:"+sentence);
						writer.write(sentenceCounter+","+sentence+"\n");
						sentenceCounter++;
					}
					
					writer.write("Answer 1:,\n");
					writer.write("Answer 2:,\n");
					
					writer.write("\n\n");
					
					writer.write("Product name:"+productNames.get(prodID)+"\n");
					int negTopic = topicID[topic] + this.number_of_topics/2;
					writer.write("Topic ID:"+ negTopic+"\n");
					writer.write("Tell me something NEGATIVE about "+ topics[posTopic].toUpperCase()+" Aspect \n");
					
					key = prodID+negTopic;
					sentences = list.get(key);
					sentenceCounter = 0;
					for(String sentence:sentences){
						writer.write(sentenceCounter+","+sentence+"\n");
						sentenceCounter++;
					}
					writer.write("Answer 1:,\n");
					writer.write("Answer 2:,\n");
					
					writer.write("\n\n");
					
				}
				
			}
			
			writer.flush();
			writer.close();
			
			
		
}
	
	
	public void doInterleaving(String category, int [] topicID){
		
		String topics [] = null;
		String productList [] = null;
		if(category.equalsIgnoreCase("camera")){
			topics = cameraTopic;
			productList = cameraProductList;
		}
		else if(category.equalsIgnoreCase("phone")){
			topics = phoneTopic;
			productList = phoneProductList;
		}
		else if(category.equalsIgnoreCase("tablet")){
			topics = tabletTopic;
			productList = tabletProductList;
		}
		else if(category.equalsIgnoreCase("tv")){
			topics = tvTopic;
			productList = tvProductList;
		}
		
		for(String prodID:productList){
			System.out.println(prodID);
			for(int topic = 0; topic<topicID.length; topic++){
			int posTopic = topicID[topic];
			int negTopic = topicID[topic] + this.number_of_topics/2;
			
			int HTSM_counter = 0;
			int HTSM_index = 0;
			int HTMM_counter = 0;
			int HTMM_index = 0;
			int ASUM_counter = 0;
			int ASUM_index = 0;
			
			int modelSequenceCounter = 0;
			
			while(HTSM_counter<2 || HTMM_counter<2 || ASUM_counter<2){
				
				int randomNUmber = 0;
				if(modelSequence.get(modelSequenceCounter).equalsIgnoreCase("HTSM"))
					randomNUmber = 0;
				else if(modelSequence.get(modelSequenceCounter).equalsIgnoreCase("HTMM"))
					randomNUmber = 1;
				else if(modelSequence.get(modelSequenceCounter).equalsIgnoreCase("ASUM"))
					randomNUmber = 2;
				
				modelSequenceCounter++;
				System.out.println("Random Number:"+randomNUmber);
				
				if(HTSM_counter<2 && randomNUmber==0){
					boolean flag = false;
					while(HTSM_index<25 && flag==false){
						System.out.print("Pos Topic:"+ posTopic+", ");
						System.out.println("HTSM Index:"+HTSM_index);
						String key = prodID+posTopic+HTSM_index;
						String sentence = HTSM_summary.get(key).replaceAll("[^a-zA-Z ]", "").toLowerCase();
						System.out.println("HTSM Sentence:"+sentence);
						
						if(!list.containsKey(prodID+posTopic)){
							
							ArrayList<String> sent = new ArrayList<String>();
							sent.add(sentence);
							list.put(prodID+posTopic, sent);
							resultWriter.write("HTSM,"+prodID+","+posTopic+","+HTSM_counter+","+sentence+"\n");
							flag = true;
							HTSM_counter++;
							break;
						}
						else {
							ArrayList<String> sentlist = list.get(prodID+posTopic);
							if(!sentlist.contains(sentence)){
								sentlist.add(sentence);
								list.put(prodID+posTopic, sentlist);
								resultWriter.write("HTSM,"+prodID+","+posTopic+","+HTSM_counter+","+sentence+"\n");
								
								flag = true;
								HTSM_counter++;
								break;
							}else{
								sentlist.add(sentence);
								list.put(prodID+posTopic, sentlist);
								resultWriter.write("HTSM,"+prodID+","+posTopic+","+HTSM_counter+","+sentence+"\n");
								
								//no break and falg set as we need two unique sentence
							}
						}
						HTSM_index++;
					}
					
				}
				
				if(HTMM_counter<2 && randomNUmber==1){
					boolean flag = false;
					while(HTMM_index<25 && flag==false){
						System.out.print("Pos Topic:"+ posTopic+", ");
						System.out.println("HTMM Index:"+HTMM_index);
						String key = prodID+posTopic+HTMM_index;
						String sentence = HTMM_summary.get(key).replaceAll("[^a-zA-Z ]", "").toLowerCase();
						System.out.println("HTMM Sentence:"+sentence);
						
						if(!list.containsKey(prodID+posTopic)){
							
							ArrayList<String> sent = new ArrayList<String>();
							sent.add(sentence);
							list.put(prodID+posTopic, sent);
							resultWriter.write("HTMM,"+prodID+","+posTopic+","+HTMM_counter+","+sentence+"\n");
							
							flag = true;
							HTMM_counter++;
							break;
						}
						else {
							//System.out.println("IN HTMM contains key :");
							ArrayList<String> sentlist = list.get(prodID+posTopic);
							if(!sentlist.contains(sentence)){
								sentlist.add(sentence);
								list.put(prodID+posTopic, sentlist);
								resultWriter.write("HTMM,"+prodID+","+posTopic+","+HTMM_counter+","+sentence+"\n");
								
								flag = true;
								HTMM_counter++;
								break;
							}else{
								sentlist.add(sentence);
								list.put(prodID+posTopic, sentlist);
								resultWriter.write("HTMM,"+prodID+","+posTopic+","+HTMM_counter+","+sentence+"\n");
								
							
							}
						}
						HTMM_index++;
					}
					
				}
				
				if(ASUM_counter<2 && randomNUmber==2){
					boolean flag = false;
					while(ASUM_index<25 && flag==false){
						System.out.print("Pos Topic:"+ posTopic+", ");
						System.out.println("ASUM Index:"+ASUM_index);
						String key = prodID+posTopic+ASUM_index;
						String sentence = ASUM_summary.get(key).replaceAll("[^a-zA-Z ]", "").toLowerCase();
						System.out.println("ASUM Sentence:"+sentence);
						
						if(!list.containsKey(prodID+posTopic)){
							
							ArrayList<String> sent = new ArrayList<String>();
							sent.add(sentence);
							list.put(prodID+posTopic, sent);
							resultWriter.write("ASUM,"+prodID+","+posTopic+","+ASUM_counter+","+sentence+"\n");
							
							flag = true;
							ASUM_counter++;
							break;
						}
						else {
							ArrayList<String> sentlist = list.get(prodID+posTopic);
							if(!sentlist.contains(sentence)){
								sentlist.add(sentence);
								list.put(prodID+posTopic, sentlist);
								resultWriter.write("ASUM,"+prodID+","+posTopic+","+ASUM_counter+","+sentence+"\n");
								
								flag = true;
								ASUM_counter++;
								break;
							}else{
								sentlist.add(sentence);
								list.put(prodID+posTopic, sentlist);
								resultWriter.write("ASUM,"+prodID+","+posTopic+","+ASUM_counter+","+sentence+"\n");
							
							}
						}
						ASUM_index++;
					}
					
					
				}
				
			}
			
		
			//Negative Topics //
			
			posTopic = negTopic = topicID[topic] + this.number_of_topics/2;
			
			HTSM_counter = 0;
			HTSM_index = 0;
			HTMM_counter = 0;
			HTMM_index = 0;
			ASUM_counter = 0;
			ASUM_index = 0;
			
			while(HTSM_counter<2 || HTMM_counter<2 || ASUM_counter<2){
				int randomNUmber = 0 + r.nextInt(3);
				if(HTSM_counter<2 && randomNUmber==0){
					boolean flag = false;
					while(HTSM_index<25 && flag==false){
						System.out.print("Neg Topic:"+ posTopic+", ");
						System.out.println("HTSM Index:"+HTSM_index);
						String key = prodID+posTopic+HTSM_index;
						String sentence = HTSM_summary.get(key).replaceAll("[^a-zA-Z ]", "").toLowerCase();
						System.out.println("HTSM Sentence:"+sentence);
						
						if(!list.containsKey(prodID+posTopic)){
							
							ArrayList<String> sent = new ArrayList<String>();
							sent.add(sentence);
							list.put(prodID+posTopic, sent);
							resultWriter.write("HTSM,"+prodID+","+posTopic+","+HTSM_counter+","+sentence+"\n");
							
							flag = true;
							HTSM_counter++;
							break;
						}
						else {
							ArrayList<String> sentlist = list.get(prodID+posTopic);
							if(!sentlist.contains(sentence)){
								sentlist.add(sentence);
								list.put(prodID+posTopic, sentlist);
								resultWriter.write("HTSM,"+prodID+","+posTopic+","+HTSM_counter+","+sentence+"\n");
								
								flag = true;
								HTSM_counter++;
								break;
							}else{
								sentlist.add(sentence);
								list.put(prodID+posTopic, sentlist);
								resultWriter.write("HTSM,"+prodID+","+posTopic+","+HTSM_counter+","+sentence+"\n");
								
							}
						}
						HTSM_index++;
					}
					
				}
				
				if(HTMM_counter<2 && randomNUmber==1){
					boolean flag = false;
					while(HTMM_index<25 && flag==false){
						System.out.print("Neg Topic:"+ posTopic+", ");
						System.out.println("HTMM Index:"+HTMM_index);
						String key = prodID+posTopic+HTMM_index;
						String sentence = HTMM_summary.get(key).replaceAll("[^a-zA-Z ]", "").toLowerCase();
						System.out.println("HTMM Sentence:"+sentence);
						
						if(!list.containsKey(prodID+posTopic)){
							
							ArrayList<String> sent = new ArrayList<String>();
							sent.add(sentence);
							list.put(prodID+posTopic, sent);
							resultWriter.write("HTMM,"+prodID+","+posTopic+","+HTMM_counter+","+sentence+"\n");
							
							flag = true;
							HTMM_counter++;
							break;
						}
						else {
							//System.out.println("IN HTMM contains key :");
							ArrayList<String> sentlist = list.get(prodID+posTopic);
							if(!sentlist.contains(sentence)){
								sentlist.add(sentence);
								list.put(prodID+posTopic, sentlist);
								resultWriter.write("HTMM,"+prodID+","+posTopic+","+HTMM_counter+","+sentence+"\n");
								
								flag = true;
								HTMM_counter++;
								break;
							}else{
								sentlist.add(sentence);
								list.put(prodID+posTopic, sentlist);
								resultWriter.write("HTMM,"+prodID+","+posTopic+","+HTMM_counter+","+sentence+"\n");
								
							}
						}
						HTMM_index++;
					}
					
				}
				
				if(ASUM_counter<2 && randomNUmber==2){
					boolean flag = false;
					while(ASUM_index<25 && flag==false){
						System.out.print("Neg Topic:"+ posTopic+", ");
						System.out.println("ASUM Index:"+ASUM_index);
						String key = prodID+posTopic+ASUM_index;
						String sentence = ASUM_summary.get(key).replaceAll("[^a-zA-Z ]", "").toLowerCase();
						System.out.println("ASUM Sentence:"+sentence);
						
						if(!list.containsKey(prodID+posTopic)){
							
							ArrayList<String> sent = new ArrayList<String>();
							sent.add(sentence);
							list.put(prodID+posTopic, sent);
							resultWriter.write("ASUM,"+prodID+","+posTopic+","+ASUM_counter+","+sentence+"\n");
							
							flag = true;
							ASUM_counter++;
							break;
						}
						else {
							ArrayList<String> sentlist = list.get(prodID+posTopic);
							if(!sentlist.contains(sentence)){
								sentlist.add(sentence);
								list.put(prodID+posTopic, sentlist);
								resultWriter.write("ASUM,"+prodID+","+posTopic+","+ASUM_counter+","+sentence+"\n");
								
								flag = true;
								ASUM_counter++;
								break;
							}else{
								sentlist.add(sentence);
								list.put(prodID+posTopic, sentlist);
								resultWriter.write("ASUM,"+prodID+","+posTopic+","+ASUM_counter+","+sentence+"\n");
								
							}
						}
						ASUM_index++;
					}
					
					
				}
				
			}
			
			
			
			
			}
			
		}
		resultWriter.flush();
		resultWriter.close();
		
	}
	
	
	public void readSummaryRealResult(String fileName){
		BufferedReader fileReader = null;
		try {
			 
			fileReader = new BufferedReader(new FileReader(fileName));
			String line;
			while ((line = fileReader.readLine()) != null) {
				//System.out.println(line);
				String infos [] = line.split(",");
				String modelName = infos[0];
				modelSequence.add(modelName);
				String productID = infos[1];
				String topicID = infos[2];
				productID = productID+topicID;
				String summary = infos[4];
				
				if(modelName.equalsIgnoreCase("HTSM")){
					if(!HTSM.containsKey(productID)){
						ArrayList<String> tmp = new ArrayList<String>();
						tmp.add(summary);
						
						HTSM.put(productID, tmp);
					}
					else{
						ArrayList<String> tmp = HTSM.get(productID);
						tmp.add(summary);
						HTSM.put(productID, tmp);
					}
					
				}else if(modelName.equalsIgnoreCase("HTMM")){
					if(!HTMM.containsKey(productID)){
						ArrayList<String> tmp = new ArrayList<String>();
						tmp.add(summary);
						
						HTMM.put(productID, tmp);
					}
					else{
						ArrayList<String> tmp = HTMM.get(productID);
						tmp.add(summary);
						HTMM.put(productID, tmp);
					}
					
				}else if(modelName.equalsIgnoreCase("ASUM")){
					if(!ASUM.containsKey(productID)){
						ArrayList<String> tmp = new ArrayList<String>();
						tmp.add(summary);
						
						ASUM.put(productID, tmp);
					}
					else{
						ArrayList<String> tmp = ASUM.get(productID);
						tmp.add(summary);
						ASUM.put(productID, tmp);
					}
					
				}
			}
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} 
	 
		
	}
	
	
	
	public void readSummarySurvey(String fileName){
		
		/*System.out.println("HTSM size"+HTSM.size());
		System.out.println("HTMM size"+HTMM.size());
		System.out.println("ASUM size"+ASUM.size());*/
		
		BufferedReader fileReader = null;
		try {
			 
			fileReader = new BufferedReader(new FileReader(fileName));
			String line;
			String productName = "";
			String topicID = "";
			while ((line = fileReader.readLine()) != null) {
				//System.out.println(line);
				if(line.contains("Product name")){
					productName = line.substring(line.indexOf(":")+1, line.indexOf(","));
				}
				
				if(line.contains("Topic ID")){
					topicID = line.substring(line.indexOf(":")+1, line.indexOf(","));
				}
				
				
				if(line.contains("Answer")){
					
					if(line.split(",").length>1){
						String summary = line.split(",")[1];
						String productID = ProductKeys.get(productName);
						productID = productID+topicID;

						/*System.out.println("Topic ID:"+topicID);
						System.out.println("Product Name:"+productName);
						System.out.println("Product ID:"+productID);
						System.out.println(summary);
						*/

						if(HTSM.get(productID).contains(summary)){
							HTSM_score = HTSM_score+ 1;
							System.out.println("sentence:" + summary+", index:"+this.number_of_Total_sentences);
							this.number_of_Total_sentences++;
						}else if(HTMM.get(productID).contains(summary)){
							HTMM_score = HTMM_score+ 1;
							System.out.println("sentence:" + summary+", index:"+this.number_of_Total_sentences);
							
							this.number_of_Total_sentences++;
						}else if(ASUM.get(productID).contains(summary)){
							ASUM_score = ASUM_score+ 1;
							System.out.println("sentence:" + summary+", index:"+this.number_of_Total_sentences);
							this.number_of_Total_sentences++;
						}
						
					}
					
				}
				
	
			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} 
	 
		
	}
	
	public void Scores(int normalizer){
		
		System.out.println("Total Sentence:"+ this.number_of_Total_sentences);
		double s = ASUM_score+HTMM_score+HTSM_score;
		System.out.println("Total:"+ s);
		System.out.println("ASUM,score,"+ASUM_score);
		System.out.println("HTMM,score,"+HTMM_score);
		System.out.println("HTSM,score,"+HTSM_score);
		
		
		System.out.format("ASUM,score, %.3f\n",ASUM_score/number_of_Total_sentences);
		System.out.format("HTMM,score, %.3f\n",HTMM_score/number_of_Total_sentences);
		System.out.format("HTSM,score, %.3f\n",HTSM_score/number_of_Total_sentences);
		
	}
	
	
	public void readcsv(String fileName, String category, String modelName)
	{
		
		BufferedReader fileReader = null;
		if(category.equalsIgnoreCase("tablet"))
			number_of_topics = 30;
		else if(category.equalsIgnoreCase("camera"))
			number_of_topics = 26;
		else if(category.equalsIgnoreCase("phone"))
			number_of_topics = 26;
		else if(category.equalsIgnoreCase("tv"))
			number_of_topics = 16;
		
		try {
	 
			fileReader = new BufferedReader(new FileReader(fileName));
			String line;
			if(modelName.equalsIgnoreCase("LRHTSM")){	
			
				while ((line = fileReader.readLine()) != null) {
					String infos[] = line.split(",");
					String prodID = infos[0];
					int topicID = Integer.parseInt(infos[1]);
					int sentenceID = Integer.parseInt(infos[2]);
					String sentence = infos[3];
					
					System.out.println();
					
					String hashKey = prodID+topicID+sentenceID;
					HTSM_summary.put(hashKey, sentence);
					
					System.out.println("HTSM:"+prodID+","+topicID+","+sentenceID);
					System.out.println(HTSM_summary.get(hashKey));
					
					//line = "";
				}
			
			}else if(modelName.equalsIgnoreCase("LRHTMM")){	
			
				while ((line = fileReader.readLine()) != null) {
					String infos[] = line.split(",");
					String prodID = infos[0];
					int topicID = Integer.parseInt(infos[1]);
					int sentenceID = Integer.parseInt(infos[2]);
					String sentence = infos[3];
					
					String hashKey = prodID+topicID+sentenceID;
					HTMM_summary.put(hashKey, sentence);
					System.out.println("HTMM:"+prodID+","+topicID+","+sentenceID);
					
					System.out.println(HTMM_summary.get(hashKey));
					
					//line = "";
				}
			
			}else if(modelName.equalsIgnoreCase("ASUM")){	
			
				
				while ((line = fileReader.readLine()) != null) {
					String infos[] = line.split(",");
					String prodID = infos[0];
					int topicID = Integer.parseInt(infos[1]);
					int sentenceID = Integer.parseInt(infos[2]);
					String sentence = infos[3];
					
					String hashKey = prodID+topicID+sentenceID;
					ASUM_summary.put(hashKey, sentence);
					System.out.println("ASUM:"+prodID+","+topicID+","+sentenceID);
					
					System.out.println(ASUM_summary.get(hashKey));
					
					//line = "";
				}
			
			}

					 
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} 
	 
	  
	}

		
	public static void main(String[] args) {
		
		summaryMeasureNew com = new summaryMeasureNew();
		String modelNames[] = {"LRHTMM","LRHTSM","ASUM"}; // 2topic, pLSA, HTMM, LRHTMM, Tensor, LDA_Gibbs, LDA_Variational, HTSM, LRHTSM
		
		int totalAnnot = 0;
		int number_of_topics = 0;
		
		String category = "phone";
		if(category.equalsIgnoreCase("tablet") || category.equalsIgnoreCase("camera") || category.equalsIgnoreCase("phone"))
			totalAnnot = 6;
		
		if(category.equalsIgnoreCase("tv"))
			totalAnnot = 3;
		
		int totalCount = 4*2*2*totalAnnot; // # of products * #number of sentiment * #of sentence * #of annotator
		
		// reading the summary result file 
		com.setCategory(category);
		String summaryFilePath = "./survey/"+category+"_DocSummary_Result.csv";
		com.readSummaryRealResult(summaryFilePath);
		
		int tablettopicID [] = {1,2,3,7};
		int phonetopicID [] = {2,3,5,8};
		int cameratopicID [] = {1,9,10,12};
		int tvtopicID [] = {0,1,2,3};
		
		if(category.equalsIgnoreCase("tablet"))
			number_of_topics = 30;
		else if(category.equalsIgnoreCase("camera"))
			number_of_topics = 26;
		else if(category.equalsIgnoreCase("phone"))
			number_of_topics = 26;
		else if(category.equalsIgnoreCase("tv"))
			number_of_topics = 16;
		
		//reading actual summary files
		for(String modelName:modelNames){
		
			summaryFilePath = "./survey/"+ modelName +"_" +category+"_Topics_" + number_of_topics + "_Summary.txt";
			com.readcsv(summaryFilePath, category, modelName);
			
		}
		
		System.out.println("Reading 1 Done");
		
		int topicID[] = null;
		
		if(category.equalsIgnoreCase("tablet"))
			topicID = tablettopicID;
		else if(category.equalsIgnoreCase("camera"))
			topicID = cameratopicID;
		else if(category.equalsIgnoreCase("phone"))
			topicID = phonetopicID;
		else if(category.equalsIgnoreCase("tv"))
			topicID = tvtopicID;
		
		String outputFileResult = "./survey/"+category+"_DocSummary_Result_new.csv";
		com.setResultWriter(outputFileResult);
		
		com.doInterleaving(category, topicID);
		System.out.println("Reading 2 Done");

		
		/*String filePath = "./survey/survey/summaryResult/";	
		for(int annotatorNumber = 1;annotatorNumber<=totalAnnot;annotatorNumber++)
			{
				System.out.println(annotatorNumber);
				String surveyFilePath = filePath+category+"_"+annotatorNumber+".csv";
				com.readSummarySurvey(surveyFilePath);
			}
		com.Scores(totalCount);
		*/
		}

	

}
