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

public class intrusionKappaMeasure1 {
	
	private String HTSM_array[][];
	private String HTMM_array[][];
	private String LDA_array[][];
	private String ASUM_array[][];
	
	private int totalWords = 0;
	private int LDAtotalWord = 0;
	private int HTSMtotalWord = 0;
	private int HTMMtotalWord = 0;
	private int ASUMtotalWord = 0;
	
	
	private int LDA_score[][]; // MP_k_m
	private int HTSM_score[][]; // MP_k_m
	private int HTMM_score[][]; // MP_k_m
	private int ASUM_score[][]; // MP_k_m
	
	private double LDA_Intra_sum = 0;
	private double LDA_Inter_sum = 0;
	private double HTSM_Intra_sum = 0;
	private double HTSM_Inter_sum = 0;
	private double HTMM_Intra_sum = 0;
	private double HTMM_Inter_sum = 0;
	private double ASUM_Intra_sum = 0;
	private double ASUM_Inter_sum = 0;
	
	
	private HashMap<String, ArrayList<String>> pairOneAns = new HashMap<String, ArrayList<String>>();
	private HashMap<String, ArrayList<String>> pairTwoAns = new HashMap<String, ArrayList<String>>();
	private int [][] booleanMatrix;
	
	private int number_of_topics= 0;
	
	public intrusionKappaMeasure1(){
		
	}
	
	public void setNumberOfTopics(int numberTopics){
		this.number_of_topics = numberTopics;
		
		LDA_score = new int [this.number_of_topics][2];
		HTSM_score = new int [this.number_of_topics][2];
		
		HTMM_score = new int [this.number_of_topics][2];
		ASUM_score = new int [this.number_of_topics][2];
		for(int k=0;k<this.number_of_topics;k++){
			LDA_score[k][0] = 0;
			LDA_score[k][1] = 0;
			
			HTSM_score[k][0] = 0;
			HTSM_score[k][1] = 0;
			
			HTMM_score[k][0] = 0;
			HTMM_score[k][1] = 0;
			
			ASUM_score[k][0] = 0;
			ASUM_score[k][1] = 0;
		}
			
		
	}
	
	public void accumulateScore(int annotator){
		for(int k =0; k<this.number_of_topics;k++){
			LDA_Intra_sum = LDA_Intra_sum+LDA_score[k][0];
			LDA_Inter_sum = LDA_Inter_sum+LDA_score[k][1];
			
			HTSM_Intra_sum = HTSM_Intra_sum+HTSM_score[k][0];
			HTSM_Inter_sum = HTSM_Inter_sum+HTSM_score[k][1];
			
			
			HTMM_Intra_sum = HTMM_Intra_sum+HTMM_score[k][0];
			HTMM_Inter_sum = HTMM_Inter_sum+HTMM_score[k][1];
			
			
			ASUM_Intra_sum = ASUM_Intra_sum+ASUM_score[k][0];
			ASUM_Inter_sum = ASUM_Inter_sum+ASUM_score[k][1];
		}
		
		System.out.println("Total Word:"+ this.totalWords);
		System.out.println("LDA Total Word:"+ this.LDAtotalWord);
		
		System.out.println("HTSM Total Word:"+ this.HTSMtotalWord);
		System.out.println("HTMM Total Word:"+ this.HTMMtotalWord);
		
		System.out.println("ASUM Total Word:"+ this.ASUMtotalWord);
		
		System.out.format("LDA,Intra score, %.3f,Inter score, %.3f\n",LDA_Intra_sum,LDA_Inter_sum);
		System.out.format("HTMM,Intra score %.3f,Inter score,%.3f\n",HTMM_Intra_sum,HTMM_Inter_sum);
		System.out.format("ASUM,Intra score %.3f,Inter score,%.3f\n",ASUM_Intra_sum,ASUM_Inter_sum);
		System.out.format("HTSM,Intra score %.3f,Inter score,%.3f\n",HTSM_Intra_sum,HTSM_Inter_sum);
		
		
		System.out.format("LDA,score, %.3f\n",(LDA_Intra_sum + LDA_Inter_sum)/this.LDAtotalWord);
		System.out.format("HTMM,score, %.3f\n",(HTMM_Intra_sum+HTMM_Inter_sum)/this.HTMMtotalWord);
		System.out.format("ASUM,score, %.3f\n",(ASUM_Intra_sum+ASUM_Inter_sum)/this.ASUMtotalWord);
		System.out.format("HTSM,score, %.3f\n",(HTSM_Intra_sum+HTSM_Inter_sum)/this.HTSMtotalWord);

		
		

		System.out.println("Precision");
		System.out.format("LDA,Intra score, %.3f,Inter score, %.3f\n",LDA_Intra_sum/this.LDAtotalWord,LDA_Inter_sum/this.LDAtotalWord);
		System.out.format("HTMM,Intra score %.3f,Inter score,%.3f\n",HTMM_Intra_sum/this.HTMMtotalWord,HTMM_Inter_sum/this.HTMMtotalWord);
		System.out.format("ASUM,Intra score %.3f,Inter score,%.3f\n",ASUM_Intra_sum/this.ASUMtotalWord,ASUM_Inter_sum/this.ASUMtotalWord);
		System.out.format("HTSM,Intra score %.3f,Inter score,%.3f\n",HTSM_Intra_sum/this.HTSMtotalWord,HTSM_Inter_sum/this.HTSMtotalWord);
		
		
		System.out.println("Recall");
		System.out.format("LDA,Intra score, %.3f,Inter score, %.3f\n",LDA_Intra_sum/(this.number_of_topics*annotator),LDA_Inter_sum/(this.number_of_topics*annotator));
		System.out.format("HTMM,Intra score %.3f,Inter score,%.3f\n",HTMM_Intra_sum/(this.number_of_topics*annotator),HTMM_Inter_sum/(this.number_of_topics*annotator));
		System.out.format("ASUM,Intra score %.3f,Inter score,%.3f\n",ASUM_Intra_sum/(this.number_of_topics*annotator),ASUM_Inter_sum/(this.number_of_topics*annotator));
		System.out.format("HTSM,Intra score %.3f,Inter score,%.3f\n",HTSM_Intra_sum/(this.number_of_topics*annotator),HTSM_Inter_sum/(this.number_of_topics*annotator));
		
		
		
		
		
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
				
		if(modelName.equalsIgnoreCase("LRHTSM"))
			HTSM_array = new String[7][this.number_of_topics];
		else if(modelName.equalsIgnoreCase("LRHTMM"))
			HTMM_array = new String[7][this.number_of_topics];
		else if(modelName.equalsIgnoreCase("LDA"))
			LDA_array = new String[7][this.number_of_topics];
		else if(modelName.equalsIgnoreCase("ASUM"))
			ASUM_array = new String[7][this.number_of_topics];
		
		try {
	 
			fileReader = new BufferedReader(new FileReader(fileName));
			String line;
			
			if(!modelName.equalsIgnoreCase("ASUM")){
			
				while ((line = fileReader.readLine()) != null) {
		 
					int topicNumber = Integer.parseInt(line.substring(line.indexOf(' ')+1,line.indexOf('(')));
					line = line.substring(line.indexOf(':')+1).replaceAll("\t", "");
					String words[] = line.split(",");
					for(int i=0; i<words.length;i++){
						if(modelName.equalsIgnoreCase("LRHTSM"))
							HTSM_array[i][topicNumber] = words[i];
						else if(modelName.equalsIgnoreCase("LRHTMM"))
							HTMM_array[i][topicNumber] = words[i];
						else if(modelName.equalsIgnoreCase("LDA"))
							LDA_array[i][topicNumber] = words[i];
						
					}
				}
			}
			else{
				
				int first = 0;
				int rowNumber = 0;
				
				while ((line = fileReader.readLine()) != null) {
					
					if(first == 0){
						first++;
						continue;
					}
						
					
					String words[] = line.split(",");
					for(int i=0; i<number_of_topics;i++){
					  ASUM_array[rowNumber][i] = words[i];
					
					}
					
					rowNumber++;
				}
			}
					 
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} 
	 
	  
	}

	public void readSurvey(String fileName, String pairFile, int part, int annotatorNumber, String category){
		
		booleanMatrix = new int [7][2]; // 7 words and 2 annotators
		
		for(int a=0;a<7;a++){
			for(int b=0;b<2;b++)
				booleanMatrix[a][b] = 0;
		}
		
		
		BufferedReader fileReader = null;
		BufferedReader pairFileReader = null;
		if(category.equalsIgnoreCase("tablet"))
			number_of_topics = 30;
		else if(category.equalsIgnoreCase("camera"))
			number_of_topics = 26;
		else if(category.equalsIgnoreCase("phone"))
			number_of_topics = 26;
		else if(category.equalsIgnoreCase("tv"))
			number_of_topics = 16;

		int startTopic = 0;
		int endTopic = number_of_topics;
		
		if(part==1){
			startTopic = 0;
			endTopic = number_of_topics/2 - 1;
 		}else{
 			startTopic = number_of_topics/2;
			endTopic = number_of_topics - 1;
 		}

		try {

			fileReader = new BufferedReader(new FileReader(fileName));
			pairFileReader =  new BufferedReader(new FileReader(pairFile));
			String line, linePair;
			int lineCounter = 0;
			while ((line = fileReader.readLine()) != null && (linePair = pairFileReader.readLine())!=null) {
				
				if(lineCounter==10){ // LDA Answer 1

					String [] words = line.split(",");
					int topicIndex = startTopic;
					for(int i = 1; i<words.length;i++){
						
						if(words[i].trim().matches("[a-zA-Z0-9]+") && words[i].trim()!=null)
						{
							int wordNo = 1;
							String key = "LDA"+topicIndex;
							if(!pairOneAns.containsKey(key))
							{
								ArrayList<String> tmp = new ArrayList<String>();
								tmp.add(words[i].trim());
								pairOneAns.put(key, tmp);
							}
							else{
								ArrayList<String> tmp = pairOneAns.get(key);
								tmp.add(words[i].trim());
								pairOneAns.put(key, tmp);
							}
						}
						
						topicIndex++;
					}
					
					
					words = linePair.split(",");
					topicIndex = startTopic;
					for(int i = 1; i<words.length;i++){
						
						if(words[i].trim().matches("[a-zA-Z0-9]+") && words[i].trim()!=null)
						{
							int wordNo = 1;
							String key = "LDA"+topicIndex;
							if(!pairTwoAns.containsKey(key))
							{
								ArrayList<String> tmp = new ArrayList<String>();
								tmp.add(words[i].trim());
								pairTwoAns.put(key, tmp);
							}
							else{
								ArrayList<String> tmp = pairTwoAns.get(key);
								tmp.add(words[i].trim());
								pairTwoAns.put(key, tmp);
							}
						}
						topicIndex++;
					}
						
						
					}
					
					
					
				else if(lineCounter==11){ // LDA Answer 2

					String [] words = line.split(",");
					int topicIndex = startTopic;
					for(int i = 1; i<words.length;i++){
						
						if(words[i].trim().matches("[a-zA-Z0-9]+") && words[i].trim()!=null)
						{
							int wordNo = 2;
							String key = "LDA"+topicIndex;
							if(!pairOneAns.containsKey(key))
							{
								ArrayList<String> tmp = new ArrayList<String>();
								tmp.add(words[i].trim());
								pairOneAns.put(key, tmp);
							}
							else{
								ArrayList<String> tmp = pairOneAns.get(key);
								tmp.add(words[i].trim());
								pairOneAns.put(key, tmp);
							}
						}
						
						topicIndex++;
					}
					
					
					words = linePair.split(",");
					topicIndex = startTopic;
					for(int i = 1; i<words.length;i++){
						
						if(words[i].trim().matches("[a-zA-Z0-9]+") && words[i].trim()!=null)
						{
							int wordNo = 2;
							String key = "LDA"+topicIndex;
							if(!pairTwoAns.containsKey(key))
							{
								ArrayList<String> tmp = new ArrayList<String>();
								tmp.add(words[i].trim());
								pairTwoAns.put(key, tmp);
							}
							else{
								ArrayList<String> tmp = pairTwoAns.get(key);
								tmp.add(words[i].trim());
								pairTwoAns.put(key, tmp);
							}
						}
						topicIndex++;
					}
						
						
					
				}else if(lineCounter==22){ // HTSM Answer 1
					

					String [] words = line.split(",");
					int topicIndex = startTopic;
					for(int i = 1; i<words.length;i++){
						
						if(words[i].trim().matches("[a-zA-Z0-9]+") && words[i].trim()!=null)
						{
							int wordNo = 1;
							String key = "HTSM"+topicIndex;
							if(!pairOneAns.containsKey(key))
							{
								ArrayList<String> tmp = new ArrayList<String>();
								tmp.add(words[i].trim());
								pairOneAns.put(key, tmp);
							}
							else{
								ArrayList<String> tmp = pairOneAns.get(key);
								tmp.add(words[i].trim());
								pairOneAns.put(key, tmp);
							}
						}
						
						topicIndex++;
					}
					
					
					words = linePair.split(",");
					topicIndex = startTopic;
					for(int i = 1; i<words.length;i++){
						
						if(words[i].trim().matches("[a-zA-Z0-9]+") && words[i].trim()!=null)
						{
							int wordNo = 1;
							String key = "HTSM"+topicIndex;
							if(!pairTwoAns.containsKey(key))
							{
								ArrayList<String> tmp = new ArrayList<String>();
								tmp.add(words[i].trim());
								pairTwoAns.put(key, tmp);
							}
							else{
								ArrayList<String> tmp = pairTwoAns.get(key);
								tmp.add(words[i].trim());
								pairTwoAns.put(key, tmp);
							}
						}
						topicIndex++;
					}
						
						
					
					
				}else if(lineCounter==23){ // HTSM Answer 2
					

					String [] words = line.split(",");
					int topicIndex = startTopic;
					for(int i = 1; i<words.length;i++){
						
						if(words[i].trim().matches("[a-zA-Z0-9]+") && words[i].trim()!=null)
						{
							int wordNo = 2;
							String key = "HTSM"+topicIndex;
							if(!pairOneAns.containsKey(key))
							{
								ArrayList<String> tmp = new ArrayList<String>();
								tmp.add(words[i].trim());
								pairOneAns.put(key, tmp);
							}
							else{
								ArrayList<String> tmp = pairOneAns.get(key);
								tmp.add(words[i].trim());
								pairOneAns.put(key, tmp);
							}
						}
						
						topicIndex++;
					}
					
					
					words = linePair.split(",");
					topicIndex = startTopic;
					for(int i = 1; i<words.length;i++){
						
						if(words[i].trim().matches("[a-zA-Z0-9]+") && words[i].trim()!=null)
						{
							int wordNo = 2;
							String key = "HTSM"+topicIndex;
							if(!pairTwoAns.containsKey(key))
							{
								ArrayList<String> tmp = new ArrayList<String>();
								tmp.add(words[i].trim());
								pairTwoAns.put(key, tmp);
							}
							else{
								ArrayList<String> tmp = pairTwoAns.get(key);
								tmp.add(words[i].trim());
								pairTwoAns.put(key, tmp);
							}
						}
						topicIndex++;
					}
						
						
				}else if(lineCounter==34){ // HTMM Answer 1

					String [] words = line.split(",");
					int topicIndex = startTopic;
					for(int i = 1; i<words.length;i++){
						
						if(words[i].trim().matches("[a-zA-Z0-9]+") && words[i].trim()!=null)
						{
							int wordNo = 1;
							String key = "HTMM"+topicIndex;
							if(!pairOneAns.containsKey(key))
							{
								ArrayList<String> tmp = new ArrayList<String>();
								tmp.add(words[i].trim());
								pairOneAns.put(key, tmp);
							}
							else{
								ArrayList<String> tmp = pairOneAns.get(key);
								tmp.add(words[i].trim());
								pairOneAns.put(key, tmp);
							}
						}
						
						topicIndex++;
					}
					
					
					words = linePair.split(",");
					topicIndex = startTopic;
					for(int i = 1; i<words.length;i++){
						
						if(words[i].trim().matches("[a-zA-Z0-9]+") && words[i].trim()!=null)
						{
							int wordNo = 1;
							String key = "HTMM"+topicIndex;
							if(!pairTwoAns.containsKey(key))
							{
								ArrayList<String> tmp = new ArrayList<String>();
								tmp.add(words[i].trim());
								pairTwoAns.put(key, tmp);
							}
							else{
								ArrayList<String> tmp = pairTwoAns.get(key);
								tmp.add(words[i].trim());
								pairTwoAns.put(key, tmp);
							}
						}
						topicIndex++;
					}
						
				
					
					
				}else if(lineCounter==35){ // HTMM Answer 2

					String [] words = line.split(",");
					int topicIndex = startTopic;
					for(int i = 1; i<words.length;i++){
						
						if(words[i].trim().matches("[a-zA-Z0-9]+") && words[i].trim()!=null)
						{
							int wordNo = 2;
							String key = "HTMM"+topicIndex;
							if(!pairOneAns.containsKey(key))
							{
								ArrayList<String> tmp = new ArrayList<String>();
								tmp.add(words[i].trim());
								pairOneAns.put(key, tmp);
							}
							else{
								ArrayList<String> tmp = pairOneAns.get(key);
								tmp.add(words[i].trim());
								pairOneAns.put(key, tmp);
							}
						}
						
						topicIndex++;
					}
					
					
					words = linePair.split(",");
					topicIndex = startTopic;
					for(int i = 1; i<words.length;i++){
						
						if(words[i].trim().matches("[a-zA-Z0-9]+") && words[i].trim()!=null)
						{
							int wordNo = 2;
							String key = "HTMM"+topicIndex;
							if(!pairTwoAns.containsKey(key))
							{
								ArrayList<String> tmp = new ArrayList<String>();
								tmp.add(words[i].trim());
								pairTwoAns.put(key, tmp);
							}
							else{
								ArrayList<String> tmp = pairTwoAns.get(key);
								tmp.add(words[i].trim());
								pairTwoAns.put(key, tmp);
							}
						}
						topicIndex++;
					}
						
					
				}else if(lineCounter==46){ // ASUM Answer 1
					

					String [] words = line.split(",");
					int topicIndex = startTopic;
					for(int i = 1; i<words.length;i++){
						
						if(words[i].trim().matches("[a-zA-Z0-9]+") && words[i].trim()!=null)
						{
							int wordNo = 1;
							String key = "ASUM"+topicIndex;
							if(!pairOneAns.containsKey(key))
							{
								ArrayList<String> tmp = new ArrayList<String>();
								tmp.add(words[i].trim());
								pairOneAns.put(key, tmp);
							}
							else{
								ArrayList<String> tmp = pairOneAns.get(key);
								tmp.add(words[i].trim());
								pairOneAns.put(key, tmp);
							}
						}
						
						topicIndex++;
					}
					
					
					words = linePair.split(",");
					topicIndex = startTopic;
					for(int i = 1; i<words.length;i++){
						
						if(words[i].trim().matches("[a-zA-Z0-9]+") && words[i].trim()!=null)
						{
							int wordNo = 1;
							String key = "ASUM"+topicIndex;
							if(!pairTwoAns.containsKey(key))
							{
								ArrayList<String> tmp = new ArrayList<String>();
								tmp.add(words[i].trim());
								pairTwoAns.put(key, tmp);
							}
							else{
								ArrayList<String> tmp = pairTwoAns.get(key);
								tmp.add(words[i].trim());
								pairTwoAns.put(key, tmp);
							}
						}
						topicIndex++;
					}
						
					
				}else if(lineCounter==47){ // ASUM Answer 2
					
					String [] words = line.split(",");
					int topicIndex = startTopic;
					for(int i = 1; i<words.length;i++){
						
						if(words[i].trim().matches("[a-zA-Z0-9]+") && words[i].trim()!=null)
						{
							int wordNo = 2;
							String key = "ASUM"+topicIndex;
							if(!pairOneAns.containsKey(key))
							{
								ArrayList<String> tmp = new ArrayList<String>();
								tmp.add(words[i].trim());
								pairOneAns.put(key, tmp);
							}
							else{
								ArrayList<String> tmp = pairOneAns.get(key);
								tmp.add(words[i].trim());
								pairOneAns.put(key, tmp);
							}
						}
						
						topicIndex++;
					}
					
					
					words = linePair.split(",");
					topicIndex = startTopic;
					for(int i = 1; i<words.length;i++){
						
						if(words[i].trim().matches("[a-zA-Z0-9]+") && words[i].trim()!=null)
						{
							int wordNo = 2;
							String key = "ASUM"+topicIndex;
							if(!pairTwoAns.containsKey(key))
							{
								ArrayList<String> tmp = new ArrayList<String>();
								tmp.add(words[i].trim());
								pairTwoAns.put(key, tmp);
							}
							else{
								ArrayList<String> tmp = pairTwoAns.get(key);
								tmp.add(words[i].trim());
								pairTwoAns.put(key, tmp);
							}
						}
						topicIndex++;
					}
						
					
				}
				lineCounter++;
			}

			// Show the scores
			

		}catch(Exception e){
			System.err.println("File "+ fileName+" Not Found!!");
			e.printStackTrace();
		}
		
		
		int normalmatrix [][] = {{0,0},{0,0}};
		
		int intramatrix [][] = {{0,0},{0,0}};
		int intermatrix [][] = {{0,0},{0,0}};
		
		
		
		// go over the real answer and check two annotators
		for(int i=startTopic; i<endTopic;i++){
			//for(int wordNo = 1;wordNo<=2;wordNo++){
			  for(int j=0; j<7;j++){
				String word = LDA_array[j][i];
				word = word.trim();
				//System.out.println("Actual Word is:"+word);
				
					String key = "LDA"+i;
					if(j>=5 && pairOneAns.containsKey(key) && pairOneAns.get(key).contains(word)){
						booleanMatrix[j][0] = 1;
						//System.out.println("Pair 1, Word no: "+ wordNo+"Word Found:"+word + " at index j:"+j);
					}
					
					if(j>=5 && pairOneAns.containsKey(key) && !pairOneAns.get(key).contains(word)){
						booleanMatrix[j][0] = 0;
						//System.out.println("Pair 1, Word no: "+ wordNo+"Word Not Found:"+word + " at index j:"+j);
					}
					
					
					if(j>=5 && pairTwoAns.containsKey(key) && pairTwoAns.get(key).contains(word)){
						booleanMatrix[j][1] = 1;
						//System.out.println("Pair 2, Word no: "+ wordNo+"Word Found:"+word + " at index j:"+j);
					}
					
					if(j>=5 && pairTwoAns.containsKey(key) && !pairTwoAns.get(key).contains(word)){
						booleanMatrix[j][1] = 0;
						//System.out.println("Pair 2, Word no: "+ wordNo+"Word Not Found:"+word + " at index j:"+j);
					}
					
					
					if(j<5 && pairOneAns.containsKey(key) && !pairOneAns.get(key).contains(word)){
						booleanMatrix[j][0] = 1;
						//System.out.println("Pair 1, Word no: "+ wordNo+" Word Not Found:"+word + " at index j:"+j);
					}
					
					/*
					if (j<5 && pairOneAns.containsKey(key) && pairOneAns.get(key).contains(word)){
						//System.out.println("Pair 1, Word no: "+ wordNo+" Word Found:"+word + " at index j:"+j);
						System.out.println("Content : "+ booleanMatrix[j][0]);
						//if(wordNo==2 && booleanMatrix[j][0]==1)
							booleanMatrix[j][0] = 0;
					}*/
					
					
					if(j<5 && pairTwoAns.containsKey(key) && !pairTwoAns.get(key).contains(word)){
						booleanMatrix[j][1] = 1;
						//System.out.println("Pair 2, Word no: "+ wordNo+" Word Not Found:"+word + " at index j:"+j);
					}
					
					/*
					if(j<5 && pairTwoAns.containsKey(key) && pairTwoAns.get(key).contains(word)){
						//System.out.println("Pair 2, Word no: "+ wordNo+" Word Found:"+word + " at index j:"+j);
						System.out.println("Content : "+ booleanMatrix[j][1]);
						//if(wordNo==2 && booleanMatrix[j][1]==1)
							booleanMatrix[j][1] = 0;
					}*/
					
					
				//}
			}
			
			//accumulate count
			for(int j=0; j<5;j++){
				normalmatrix[booleanMatrix[j][0]][booleanMatrix[j][1]]+=1;
			}
			
			int j=5;
			intramatrix[booleanMatrix[j][0]][booleanMatrix[j][1]]+=1;
			j=6;
			intermatrix[booleanMatrix[j][0]][booleanMatrix[j][1]]+=1;
			//have to clear the boolean Matrix for next topic
			
			for(int a=0;a<7;a++){
				for(int b=0;b<2;b++){
					booleanMatrix[a][b] = 0;
				}
			}
			
		
			
			for(j=0; j<7;j++){
				String word = HTSM_array[j][i];
				word = word.trim();
				for(int wordNo = 1;wordNo<=2;wordNo++){
					String key = "HTSM"+wordNo+i;
					if(j>=5 && pairOneAns.containsKey(key) && pairOneAns.get(key).contains(word)){
						booleanMatrix[j][0] = 1;
					}
					
					
					if(j>=5 && pairTwoAns.containsKey(key) && pairTwoAns.get(key).contains(word)){
						booleanMatrix[j][1] = 1;
					}
					
					
					if(j<5 && pairOneAns.containsKey(key) && !pairOneAns.get(key).contains(word)){
						booleanMatrix[j][0] = 1;
					}
					
					
					if(j<5 && pairTwoAns.containsKey(key) && !pairTwoAns.get(key).contains(word)){
						booleanMatrix[j][1] = 1;
					}
				}
			}
			
			//accumulate count
			for(j=0; j<5;j++){
				normalmatrix[booleanMatrix[j][0]][booleanMatrix[j][1]]+=1;
			}
			
			j=5;
			intramatrix[booleanMatrix[j][0]][booleanMatrix[j][1]]+=1;
			j=6;
			intermatrix[booleanMatrix[j][0]][booleanMatrix[j][1]]+=1;
			//have to clear the boolean Matrix for next topic
			
			for(int a=0;a<7;a++){
				for(int b=0;b<2;b++)
					booleanMatrix[a][b] = 0;
			}
			
			
			
			for(j=0; j<7;j++){
				String word = HTMM_array[j][i];
				word = word.trim();
				for(int wordNo = 1;wordNo<=2;wordNo++){
					String key = "HTMM"+wordNo+i;
					if(j>=5 && pairOneAns.containsKey(key) && pairOneAns.get(key).contains(word)){
						booleanMatrix[j][0] = 1;
					}
					
					
					if(j>=5 && pairTwoAns.containsKey(key) && pairTwoAns.get(key).contains(word)){
						booleanMatrix[j][1] = 1;
					}
					
					
					if(j<5 && pairOneAns.containsKey(key) && !pairOneAns.get(key).contains(word)){
						booleanMatrix[j][0] = 1;
					}
					
					
					if(j<5 && pairTwoAns.containsKey(key) && !pairTwoAns.get(key).contains(word)){
						booleanMatrix[j][1] = 1;
					}
				}
			}
			
			//accumulate count
			for(j=0; j<5;j++){
				normalmatrix[booleanMatrix[j][0]][booleanMatrix[j][1]]+=1;
			}
			
			j=5;
			intramatrix[booleanMatrix[j][0]][booleanMatrix[j][1]]+=1;
			j=6;
			intermatrix[booleanMatrix[j][0]][booleanMatrix[j][1]]+=1;
			//have to clear the boolean Matrix for next topic
			
			for(int a=0;a<7;a++){
				for(int b=0;b<2;b++)
					booleanMatrix[a][b] = 0;
			}
			
			
			
			
			
			for(j=0; j<7;j++){
				String word = ASUM_array[j][i];
				word = word.trim();
				for(int wordNo = 1;wordNo<=2;wordNo++){
					String key = "ASUM"+wordNo+i;
					if(j>=5 && pairOneAns.containsKey(key) && pairOneAns.get(key).contains(word)){
						booleanMatrix[j][0] = 1;
					}
					
					
					if(j>=5 && pairTwoAns.containsKey(key) && pairTwoAns.get(key).contains(word)){
						booleanMatrix[j][1] = 1;
					}
					
					
					if(j<5 && pairOneAns.containsKey(key) && !pairOneAns.get(key).contains(word)){
						booleanMatrix[j][0] = 1;
					}
					
					
					if(j<5 && pairTwoAns.containsKey(key) && !pairTwoAns.get(key).contains(word)){
						booleanMatrix[j][1] = 1;
					}
				}
			}
			
			//accumulate count
			for(j=0; j<5;j++){
				normalmatrix[booleanMatrix[j][0]][booleanMatrix[j][1]]+=1;
			}
			
			j=5;
			intramatrix[booleanMatrix[j][0]][booleanMatrix[j][1]]+=1;
			j=6;
			intermatrix[booleanMatrix[j][0]][booleanMatrix[j][1]]+=1;
			//have to clear the boolean Matrix for next topic
			
			for(int a=0;a<7;a++){
				for(int b=0;b<2;b++)
					booleanMatrix[a][b] = 0;
			}
			
			
			
			
		}// topic loop
		
		
		for(int a=0;a<2;a++){
			for(int b=0;b<2;b++){
				System.out.print(normalmatrix[a][b]+",");
			}
		}
		
		System.out.println();
		
		for(int a=0;a<2;a++){
			for(int b=0;b<2;b++){
				System.out.print(intramatrix[a][b]+",");
			}
		}
		System.out.println();
		
		for(int a=0;a<2;a++){
			for(int b=0;b<2;b++){
				System.out.print(intermatrix[a][b]+",");
			}
		}
		System.out.println();
		
		// normal kappa
		int total = normalmatrix[0][0] + normalmatrix[0][1] + normalmatrix[1][0] +  normalmatrix[1][1] ;
		double p_o = (double)(normalmatrix[0][0] + normalmatrix[1][1])/total;
		int firstRowSum = normalmatrix[0][0] + normalmatrix[0][1];
		int secondRowSum = normalmatrix[1][0] + normalmatrix[1][1];
		
		int firstColSum = normalmatrix[0][0] + normalmatrix[1][0];
		int secondColSum = normalmatrix[0][1] + normalmatrix[1][1];
		
		double A_no_p = (double) firstRowSum / total;
		double A_yes_p = (double) secondRowSum / total;
		
		double B_no_p = (double) firstColSum / total;
		double B_yes_p = (double) secondColSum / total;
		double p_e = (A_yes_p*B_yes_p) + (A_no_p*B_no_p);
		
		double normal_kappa = (p_o - p_e)/(1 - p_e);
		System.out.print("Normal Kappa,"+normal_kappa+"\n");
		
		
		
		// intra kappa
		total = intramatrix[0][0] + intramatrix[0][1] + intramatrix[1][0] +  intramatrix[1][1] ;
		p_o = (double)(intramatrix[0][0] + intramatrix[1][1])/total;
		firstRowSum = intramatrix[0][0] + intramatrix[0][1];
		secondRowSum = intramatrix[1][0] + intramatrix[1][1];

		firstColSum = intramatrix[0][0] + intramatrix[1][0];
		secondColSum = intramatrix[0][1] + intramatrix[1][1];

		A_no_p = (double) firstRowSum / total;
		A_yes_p = (double) secondRowSum / total;

		B_no_p = (double) firstColSum / total;
		B_yes_p = (double) secondColSum / total;
		p_e = (A_yes_p*B_yes_p) + (A_no_p*B_no_p);

		double intra_kappa = (p_o - p_e)/(1 - p_e);
		System.out.print("Intra Kappa,"+intra_kappa+"\n");

		
		
		// inter kappa
		total = intermatrix[0][0] + intermatrix[0][1] + intermatrix[1][0] +  intermatrix[1][1] ;
		p_o = (double)(intermatrix[0][0] + intermatrix[1][1])/total;
		firstRowSum = intermatrix[0][0] + intermatrix[0][1];
		secondRowSum = intermatrix[1][0] + intermatrix[1][1];

		firstColSum = intermatrix[0][0] + intermatrix[1][0];
		secondColSum = intermatrix[0][1] + intermatrix[1][1];

		A_no_p = (double) firstRowSum / total;
		A_yes_p = (double) secondRowSum / total;

		B_no_p = (double) firstColSum / total;
		B_yes_p = (double) secondColSum / total;
		p_e = (A_yes_p*B_yes_p) + (A_no_p*B_no_p);

		double inter_kappa = (p_o - p_e)/(1 - p_e);
		System.out.print("Inter Kappa,"+inter_kappa+"\n");
		
		
		
		
	}
	
		
	public static void main(String[] args) {
		
		intrusionKappaMeasure1 com = new intrusionKappaMeasure1();
		
		String modelNames[] = {"LRHTMM","LRHTSM","LDA","ASUM"}; // 2topic, pLSA, HTMM, LRHTMM, Tensor, LDA_Gibbs, LDA_Variational, HTSM, LRHTSM
		String category = "tablet";
		int number_of_topics = 0;
		int partLimit = 2;
		
		if(category.equalsIgnoreCase("tablet"))
		{
			number_of_topics = 30;
		}
		else if(category.equalsIgnoreCase("camera"))
		{
			number_of_topics = 26;
		}
		else if(category.equalsIgnoreCase("phone"))
		{
			number_of_topics = 26;
		}
		else if(category.equalsIgnoreCase("tv"))
		{
			number_of_topics = 16;
			partLimit = 1;
		}
		
		// reading the answer first
		for(String modelName:modelNames){
			String wordIntrusionFilePath = "./survey/"+ modelName +"_" +category+"_Topics_" + number_of_topics + "_WordIntrusion.txt";
			com.readcsv(wordIntrusionFilePath, category, modelName);
			
		}
		
		String filePath = "./survey/survey/result/";
		com.setNumberOfTopics(number_of_topics);
		
		int p=1; int s=1;
		
		for(int part = p;part <=p; part++){
			
			for(int annotatorNumber = s;annotatorNumber<=s;annotatorNumber++)
			{
				String mainName = filePath+category+"_"+part+"_"+annotatorNumber+".csv";
				int pairNumber = (annotatorNumber%3)+1;
				String pairFile = filePath+category+"_"+part+"_"+pairNumber+".csv";
				System.out.println(category+"_"+part+"_"+annotatorNumber+","+category+"_"+part+"_"+pairNumber);
				com.readSurvey(mainName, pairFile, part, annotatorNumber, category);
			}
		
		}
		//com.accumulateScore(3);
				
      }

	

}
