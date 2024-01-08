import java.util.Scanner;

public class Array2D {

	public static void main(String[] args) {
		int [][]a={{1,2,3},{4,5,6},{7,8,9}};//uniform init array
		//in java every row can have different number of columns there arrays are called jagged arrays
		int[][] b= {{1,2,3,4},{5,6},{7,8,9}};//jagged array
		int [][]c = new int[][] {{1,2,3},{4,5,6},{7,8,9}};
		
		
		//uninitialized array
		int[][] d=new int[3][];
		d[0] = new int[3];
		d[1] = new int[5];
		d[2] = new int[2];
		
		Scanner sc = new Scanner(System.in);
		
		System.out.println("Enter elements for the array");
		//scan array
		for (int i=0;i<d.length;i++) {
			for (int j=0;j<d[i].length;j++) {
				d[i][j]=sc.nextInt();
			}
		}
		//print array
		for (int i=0;i<d.length;i++) {
			for (int j=0;j<d[i].length;j++) {
				System.out.print(d[i][j]+" ");
			}
			System.out.println();
		}
		
		//print array using for each loop
		for(int []temp:d) {
			for (int ele:temp)
				System.out.print(ele+" ");
			System.out.println();
		}
		
		
	}

}
