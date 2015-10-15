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
import java.util.concurrent.locks.ReentrantReadWriteLock.ReadLock;
import java.util.Set;

import cc.mallet.topics.TopicModelDiagnostics.TopicScores;

public class transitionMatrxiCalculator {
	
	Map<String, Double> dictionary = new HashMap<String, Double>();
	Map<String, Integer> sortedMap;
	
	HashMap<Integer, ArrayList<Double>> prosList = new HashMap<Integer, ArrayList<Double>>();
	HashMap<Integer, ArrayList<Double>> consList = new HashMap<Integer, ArrayList<Double>>();
	
	ArrayList<String> nodeName = new ArrayList<String>();
	
	int nodes = 0;
	
	double matrix[][];
	private String cameraTopic[]= {"battery_P", "screen_P","viewfinder_P","record_P","mode_P","shutter_P","memori_P","zoom_P","len_P","picture_P","price_P","Instruction_P","video_P","battery_N", "screen_N","viewfinder_N","record_N","mode_N","shutter_N","memori_N","zoom_N","len_N","picture_N","price_N","Instruction_N","video_N"};
	private String phoneTopic[]= {"screen_P","app_P","price_P","battery_P","sound_P","camera_P","storage_P","call_P","servic_P","cpu_P","keyboard_P","design_P","text_P","screen_N","app_N","price_N","battery_N","sound_N","camera_N","storage_N","call_N","servic_N","cpu_N","keyboard_N","design_N","text_N"};
	private String tabletTopic[]= {"screen_P","app_P","price_P","battery_P","sound_P","camera_P","cpu_P","microsd_P","internet_P","keyboard_P","bluetooth_P","usb_P","gps_P","servic_P","mous_P","screen_N","app_N","price_N","battery_N","sound_N","camera_N","cpu_N","microsd_N","internet_N","keyboard_N","bluetooth_N","usb_N","gps_N","servic_N","mous_N"};
	private String tvTopic[]= {"screen_P","price_P","sound_P","servic_P","picture_P","quality_P","app_P","connection_P","screen_N","price_N","sound_N","servic_N","picture_N","quality_N","app_N","connection_N"};
	
	
	
	public void readcsv(String path,int numTopics, String category)
	{
		String [] topics = null;
		if(category.equalsIgnoreCase("camera"))
			topics = cameraTopic;
		else if(category.equalsIgnoreCase("phone"))
			topics = phoneTopic;
		else if(category.equalsIgnoreCase("tablet"))
			topics = tabletTopic;
		else if(category.equalsIgnoreCase("tv"))
			topics = tvTopic;
		matrix = new double [numTopics][numTopics+1];
		BufferedReader br_real = null;
		String line = "";
		int lineNumber = 0;
		int edgeCounter = 0;
		try {
	 
			nodeName.add("start");
			br_real = new BufferedReader(new FileReader(path));
			
			while ((line = br_real.readLine()) != null) {
				String [] numbers = line.split(",");
				for(int i=0; i<numbers.length; i++){
					if(i==0){
						matrix[lineNumber][i] = Double.parseDouble(numbers[i])>0.05?Double.parseDouble(numbers[i]):0.0;
						if(matrix[lineNumber][i]>0.02){
							edgeCounter++;
							if(!nodeName.contains(topics[lineNumber])){
								nodeName.add(topics[lineNumber]);
							}
						}
						
					
					}
					else{
						matrix[lineNumber][i] = Double.parseDouble(numbers[i])>0.1?Double.parseDouble(numbers[i]):0.0;
					}
					
					if(matrix[lineNumber][i]>0.1){
						edgeCounter++;
						if(!nodeName.contains(topics[lineNumber])){
							nodeName.add(topics[lineNumber]);
						}
						
						if(i<numTopics && !nodeName.contains(topics[i])){
							nodeName.add(topics[i]);
						}
					}
					//System.out.print(matrix[lineNumber][i]+",");
					
				}
				System.out.println();
				lineNumber++;
			}

			nodes = edgeCounter/2;
	 
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} 
	 }

	
	public void writeSVGfile(String category, int numtopics){
		String [] topics = null;
		if(category.equalsIgnoreCase("camera"))
			topics = cameraTopic;
		else if(category.equalsIgnoreCase("phone"))
			topics = phoneTopic;
		else if(category.equalsIgnoreCase("tablet"))
			topics = tabletTopic;
		else if(category.equalsIgnoreCase("tv"))
			topics = tvTopic;
		
		
		
		
		String path = "./survey/transitions/"+category+"_plot.txt";
			try{
				PrintWriter writer = new PrintWriter(new File(path));
				writer.println("digraph finite_state_machine {\n\trankdir=TB;\n\tsize=\"10,7\"\n\tnode [shape = circle];");
				
				System.out.println("Size_"+nodeName.size());
				for(int x=0; x<nodeName.size();){
					writer.write("\t{rank=same;"); 
					for(int y = 1; x<nodeName.size() && y<=5; y++)
					{
						writer.write(" "+nodeName.get(x));
						x++;
					}
					writer.write(";}\n");
				}
				
			
				
				for(int i=0; i<numtopics;i++){
					if(matrix[i][0]!=0.0){
						writer.format("\tstart -> %s [ label = \"%.3f\" ];\n",topics[i],matrix[i][0]);
					}
				}
				
				for(int i=0; i<numtopics;i++){
					for(int j=1; j<numtopics;j++){
						if(matrix[i][j]!=0.0){
							writer.format("\t%s -> %s [ label = \"%.3f\" ];\n",topics[i] ,topics[j], matrix[i][j]);
						}
					}
				}
				
				writer.println("}");

				writer.flush();
				writer.close();
				
			}
			catch(Exception e){
				System.err.println("File_"+path+" Not Found!!!");
				e.printStackTrace();
			}
		
		
		
		
	}
	
	
	
	public static void main(String[] args) {
		
		transitionMatrxiCalculator com = new transitionMatrxiCalculator();
		
		int number_of_topics = 0;
		
		
		String category = "tablet";
			if(category.equalsIgnoreCase("tablet"))
				number_of_topics = 30;
			else if(category.equalsIgnoreCase("camera"))
				number_of_topics = 26;
			else if(category.equalsIgnoreCase("phone"))
				number_of_topics = 26;
			else if(category.equalsIgnoreCase("tv"))
				number_of_topics = 16;
				String filename = "./survey/transitions/"+category+".csv";
		        com.readcsv(filename, number_of_topics, category);
		        com.writeSVGfile(category, number_of_topics);
		
	}

	

}
