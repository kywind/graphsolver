#include <bits/stdc++.h>
using namespace std;
int refn[1 << 20] , acen[1 << 20] , amin[7][1 << 20];
float dis[20][20] , vcen[1 << 20] , vmin[7][1 << 20];
int main() {
	int n , p , i , j , k , x;
	float z;
	scanf("%d%d" , &n , &p);
	for(i = 0 ; i < n ; i ++ )
		for(j = 0 ; j < n ; j ++ )
			scanf("%f" , &z) , dis[i][j] = z;
	for(i = 0 ; i < n ; i ++ ) refn[1 << i] = i;
	memset(vcen , 0x7f , sizeof(vcen));
	memset(vmin , 0x7f , sizeof(vmin));
	for(i = 1 ; i < (1 << n) ; i ++ ) {
		for(j = i ; j ; j -= j & -j) {
			z = 0;
			for(k = i ; k ; k -= k & -k)
				if(z < dis[refn[j & -j]][refn[k & -k]])
					z = dis[refn[j & -j]][refn[k & -k]];
			if(vcen[i] > z)
				vcen[i] = z , acen[i] = refn[j & -j];
		}
	}
	vmin[0][0] = 0;
	for(i = 1 ; i <= p ; i ++ )
		for(j = 0 ; j < (1 << n) ; j ++ )
			for(k = j ; k ; k = (k - 1) & j)
				if(vmin[i][j] > max(vmin[i - 1][j - k] , vcen[k]))
					vmin[i][j] = max(vmin[i - 1][j - k] , vcen[k]) , amin[i][j] = k;
	printf("%f" , vmin[p][(1 << n) - 1]);
	for(i = p , j = (1 << n) - 1 ; i ; j -= amin[i][j] , i -- ) {
		x = amin[i][j];
		printf("\n%f %d" , vcen[x] , acen[x]);
		for(k = x ; k ; k -= k & -k)
			printf(" %d" , refn[k & -k]);
	}
	return 0;
}
