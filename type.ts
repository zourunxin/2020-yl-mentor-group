interface PkgInfo {
    inDegree: number;
    outDegree: number;
    textRank: number;
    pageRank: {
        keyword: string;
        value: number;
    }[];
}
