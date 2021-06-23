#include <bits/stdc++.h>
#define MAXN ((1 << 26) + 10)
using namespace std;
bool adj[60][60];
int reff[MAXN], cnt[MAXN], max_size[MAXN], argmax_size[MAXN];
int link1[30], link2[30], link3[30];
int link4[MAXN], link5[MAXN], link6[MAXN];

void link_edge(int x) {
	static char str[110];
	cin.getline(str, 110);
	int i, len = strlen(str), y = 0;
	for(i = 0 ; i <= len ; i ++ ) {
		if(isdigit(str[i])) y = 10 * y + str[i] - '0';
		else adj[x][y - 1] = adj[y - 1][x] = 1 , y = 0;
	}
}

void init_link(int m1, int m2) {
	int i, j;
	for(i = 0 ; i < m1 ; i ++ )
		for(j = 0 ; j < m1 ; j ++ )
			if(adj[i][j])
				link1[i] |= (1 << j);
	for(i = 0 ; i < m1 ; i ++ )
		for(j = 0 ; j < m2 ; j ++ )
			if(adj[i][j + m1])
				link2[i] |= (1 << j);
	for(i = 0 ; i < m2 ; i ++ )
		for(j = 0 ; j < m2 ; j ++ )
			if(adj[i + m1][j + m1])
				link3[i] |= (1 << j);
	for(i = 1 ; i < (1 << m1) ; i ++ ) link4[i] = link4[i - (i & -i)] | link1[reff[i & -i]];
	for(i = 1 ; i < (1 << m1) ; i ++ ) link5[i] = link5[i - (i & -i)] | link2[reff[i & -i]];
	for(i = 1 ; i < (1 << m2) ; i ++ ) link6[i] = link6[i - (i & -i)] | link3[reff[i & -i]];
}

int main() {
	int n, _, i, j, k, m1, m2, flag, mask, max_ans = 0, argmax_ans = 0;
	scanf("%d %d\n", &n, &_);
	for(i = 0; i < n; i ++ )
		link_edge(i);
	m1 = n >> 1, m2 = (n + 1) >> 1;
	for(i = 1 ; i < (1 << m2) ; i ++ ) cnt[i] = cnt[i - (i & -i)] + 1;
	for(i = 0 ; i < m2 ; i ++ ) reff[1 << i] = i;
	init_link(m1, m2);
	for(i = 0 ; i < (1 << m1) ; i ++ )
		if(!(link4[i] & i) && max_size[link5[i]] < cnt[i])
			max_size[link5[i]] = cnt[i] , argmax_size[link5[i]] = i;
	for(i = 1 ; i < (1 << m2) ; i ++ )
		for(j = i ; j ; j -= (j & -j))
			if(max_size[i] < max_size[i - (j & -j)])
				max_size[i] = max_size[i - (j & -j)] , argmax_size[i] = argmax_size[i - (j & -j)];
	for(i = 0 ; i < (1 << m2) ; i ++ )
		if(!(link6[i] & i) && max_ans < cnt[i] + max_size[(~i) & ((1 << m2) - 1)])
			max_ans = cnt[i] + max_size[(~i) & ((1 << m2) - 1)] , argmax_ans = i;
	for(i = argmax_size[(~argmax_ans) & ((1 << m2) - 1)] ; i ; i -= (i & -i)) printf("%d " , reff[i & -i] + 1);
	for(i = argmax_ans ; i ; i -= (i & -i)) printf("%d " , m1 + reff[i & -i] + 1);
	return 0;
}
