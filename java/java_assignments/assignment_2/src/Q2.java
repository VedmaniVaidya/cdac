import java.util.Scanner;

public class Q2 {

	public static void main(String[] args) {
	Scanner sc = new Scanner(System.in);
	int[][] arr	= get_arr();
	System.out.println("Original:\n");
	disp_arr(arr);
	int[][] transpose_arr=transpose(arr);
	System.out.println("Transpose:");
	disp_arr(transpose_arr);
//	int[][] arr2 = get_arr();
//	disp_arr(arr2);
	int[][] arr_add = add_arr(transpose_arr, arr);
	System.out.println("Addition");
	disp_arr(arr_add);
	}
	

	public static int[][] transpose(int[][] arr_in){
		int[][] arr=new int[arr_in[0].length][arr_in.length];
		for(int i=0;i<arr_in.length;i++) {
			for(int j=0;j<arr_in[i].length;j++) {
				arr[j][i]=arr_in[i][j];
			}
		}
		
		return arr;
		
	}
	
	public static int[][] get_arr(){
		Scanner sc = new Scanner(System.in);
		int[][] arr	= new int[3][];
		for (int i=0;i<arr.length;i++) {
			arr[i] = new int[3];
		}
		
		for(int i=0;i<arr.length;i++) {
			for(int j=0;j<arr[0].length;j++) {
				arr[i][j]=sc.nextInt();
			}
		}
		return arr;
	}
	
	
	public static void disp_arr(int[][] arr) {
		for (int[] temp:arr) {
			for (int val:temp)
				System.out.print(val+" ");
			System.out.println();
		}
	}
	
	public static int[][] add_arr(int[][]arr1,int[][]arr2){
		int[][] arr = new int[3][3];
		for(int i=0;i<arr1.length;i++) {
			for(int j=0;j<arr1[i].length;j++) {
				arr[i][j]=arr1[i][j]+arr2[i][j];
			}
		}
		return arr;
		
	}
}
