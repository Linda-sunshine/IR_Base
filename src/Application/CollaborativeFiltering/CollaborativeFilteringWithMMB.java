package Application.CollaborativeFiltering;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

import structures._User;


public class CollaborativeFilteringWithMMB extends CollaborativeFiltering {
	protected double[][] m_B;
	
	public CollaborativeFilteringWithMMB(ArrayList<_User> users, int fs, int k) {
		super(users, fs, k);
	}
	
	// load B_0 and B_1 and calculate the MLE of B
	public void calculateMLEB(String fileB0, String fileB1){
		double b = 0;
		// store B_0
		double[][] m_B_0 = loadBFile(fileB0);
		// store B_1 first, later on store MLE_B
		m_B = loadBFile(fileB1);
		if(m_B_0.length != m_B.length){
			System.out.println("[Error]The dimension of B_0 and B_1 does not match!");
			return;
		}
		int kBar = m_B.length;
		m_featureSize = kBar;
		for(int i=0; i<kBar; i++){
			for(int j=i; j<kBar; j++){
				if(m_B[i][j] == 0) continue;
				b = m_B[i][j] / (m_B[i][j] + m_B_0[i][j]);
				m_B[i][j] = b;
				m_B[j][i] = b;
			}
		}
	}
	@Override
	// calculate the similarity between two users based on the product of mixture and group affinity.
	protected double calculateSimilarity(double[] ui, double[] uj){
		if(ui.length != m_B.length)
			System.out.println("Wrong dimension of user mixture!");
		double sim = 0;
		for(int i=0; i<ui.length; i++){
			for(int j=0; j<uj.length; j++){
				sim += ui[i]*uj[j]*m_B[i][j];
			}
		}
		return sim;
	}
	
	// load group affinity file
	public double[][] loadBFile(String fileName){
		int kBar = -1;
		double[][] weights = null;
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(fileName), "UTF-8"));
			String line;
			int index = 0;
			while((line = reader.readLine()) != null){
				String[] ws = line.split("\t");
				// get the kBar
				if(kBar == -1 && ws.length > 0){
					kBar = ws.length;
					weights = new double[kBar][kBar];
				}
				if(ws.length != kBar)
					System.out.println("[error]The dimension of B does not match kBar!");
				else{
					for(int i=0; i<ws.length; i++){
						weights[index][i] = Double.valueOf(ws[i]);
					}
					index++;
				}
			}
			reader.close();
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", fileName);
			e.printStackTrace();
		}
		System.out.println("[Info]Finish loading B file " + fileName);
		return weights;
	}
	

}
