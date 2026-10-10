-- 帖子互动指标表：由 aiotieba 的主题帖接口全量扫描写入。

CREATE TABLE IF NOT EXISTS analysis.thread_engagement (
    tid             bigint PRIMARY KEY,
    fid             bigint      NOT NULL,
    -- 必须是显式传入，不设默认值
    view_num        integer,          -- 浏览量
    agree_num       integer,          -- 点赞
    disagree_num    integer,          -- 踩
    share_num       integer,          -- 分享
    reply_num_live  integer,          -- 抓取时刻回复数（可与库内比对）
    is_good         boolean,          -- 精华帖（吧务人工加精）= 官方质量标注
    is_top          boolean,          -- 置顶
    is_share        boolean,          -- 转发帖
    author_id       bigint,
    create_time     timestamptz,      -- 服务端时间戳
    last_time       timestamptz,
    scraped_at      timestamptz NOT NULL DEFAULT now(),
    source          text,
    parser_version  text
);
-- 幂等迁移
ALTER TABLE analysis.thread_engagement ADD COLUMN IF NOT EXISTS is_good     boolean;
ALTER TABLE analysis.thread_engagement ADD COLUMN IF NOT EXISTS is_top      boolean;
ALTER TABLE analysis.thread_engagement ADD COLUMN IF NOT EXISTS is_share    boolean;
ALTER TABLE analysis.thread_engagement ADD COLUMN IF NOT EXISTS author_id   bigint;
ALTER TABLE analysis.thread_engagement ADD COLUMN IF NOT EXISTS create_time timestamptz;
ALTER TABLE analysis.thread_engagement ADD COLUMN IF NOT EXISTS last_time   timestamptz;

CREATE INDEX IF NOT EXISTS idx_te_fid   ON analysis.thread_engagement (fid);
CREATE INDEX IF NOT EXISTS idx_te_good  ON analysis.thread_engagement (is_good) WHERE is_good;
CREATE INDEX IF NOT EXISTS idx_te_ct    ON analysis.thread_engagement (create_time);

-- 抓取进度表：记录已扫描到的页码，支持断点续跑
CREATE TABLE IF NOT EXISTS analysis.engagement_scan_progress (
    fid         bigint NOT NULL,
    sort_mode   text   NOT NULL,
    page        integer NOT NULL,
    n_threads   integer NOT NULL,
    scanned_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (fid, sort_mode, page)
);

-- 覆盖度监控。按整库统计，不绑定任何单一吧。
DROP VIEW IF EXISTS analysis.v_engagement_coverage;

CREATE VIEW analysis.v_engagement_coverage AS
SELECT
    (SELECT count(*) FROM public.thread)                        AS total_threads,
    (SELECT count(*) FROM analysis.thread_engagement)           AS scraped_threads,
    (SELECT count(*) FROM analysis.thread_engagement
       WHERE view_num IS NOT NULL)                              AS with_view,
    (SELECT count(*) FROM analysis.thread_engagement
       WHERE agree_num > 0)                                     AS with_agree,
    (SELECT count(*) FROM analysis.thread_engagement
       WHERE share_num > 0)                                     AS with_share,
    (SELECT count(*) FROM analysis.thread_engagement
       WHERE is_good)                                           AS n_good,
    (SELECT count(*) FROM analysis.thread_engagement
       WHERE is_top)                                            AS n_top,
    (SELECT count(*) FROM analysis.thread_engagement te
       JOIN public.thread t ON t.tid = te.tid)                  AS matched_to_db,
    (SELECT max(page) FROM analysis.engagement_scan_progress)   AS max_page_scanned,
    (SELECT max(scraped_at) FROM analysis.thread_engagement)    AS last_scrape;
