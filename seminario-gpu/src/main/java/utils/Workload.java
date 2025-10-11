package utils;

public enum Workload {
    A("A", 763840L, 21700L, 30256L, 26846L, 32),
    B("B", 6308736L, 89460L, 165984L, 111006L, 64),
    C("C", 51271424L, 363220L, 1016000L, 451358L, 128),
    D("D", 413396480L, 1463700L, 6865280L, 1820190L, 256),
    E("E", 3320110080L, 5876500L, 49840896L, 7310366L, 512),
    F("F", 26612709376L, 23549460L, 378340864L, 29300766L, 1024),
    G("G", 213109141504L, 94284820L, 2945059840L, 117321758L, 2048),
    H("H", 1705703301120L, 377313300L, 23233566720L, 469524510L, 4096);

    private final String name;
    private final long checksum1;
    private final long checksum2;
    private final long checksum3;
    private final long checksum4;
    private final int n;

    Workload(String name, long checksum1, long checksum2, long checksum3, long checksum4, int n) {
        this.name = name;
        this.checksum1 = checksum1;
        this.checksum2 = checksum2;
        this.checksum3 = checksum3;
        this.checksum4 = checksum4;
        this.n = n;
    }

    public String getName() { return name; }
    public long getChecksum1() { return checksum1; }
    public long getChecksum2() { return checksum2; }
    public long getChecksum3() { return checksum3; }
    public long getChecksum4() { return checksum4; }
    public int getN() { return n; }

    public static Workload fromName(String name) {
        for (Workload w : values()) {
            if (w.name.equalsIgnoreCase(name)) {
                return w;
            }
        }
        throw new IllegalArgumentException("Workload inválido: " + name);
    }
}