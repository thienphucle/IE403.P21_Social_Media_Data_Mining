## Data Description - Mô tả Dữ liệu
- *user_name*: Tên người dùng đăng video trên TikTok.

- *user_followers*: Số lượng người theo dõi kênh người dùng.

- *vid_id*: Mã định danh duy nhất của video TikTok.

- *vid_caption*: Nội dung mô tả hoặc chú thích của video.

- *vid_posttime*: Thời điểm video được đăng tải.

- *vid_scrapetime*: Thời điểm hệ thống thu thập dữ liệu của video.

- *vid_nview*: Số lượt xem video.

- *vid_nlike*: Số lượt thích video.

- *vid_ncomment*: Số lượt bình luận video.

- *vid_nshare*: Số lượt chia sẻ video.

- *vid_nsave*: số lượt lưu video.

- *vid_hashtags*: Danh sách hashtag có trong caption của video, phân tách bằng dấu phẩy.

- *vid_duration*: Độ dài của video TikTok.

- *vid_url*: Đường dẫn URL đến video gốc trên TikTok.

- *music_id*: Mã định danh của âm thanh được sử dụng trong video.

- *music_title*: Tên của âm thanh hoặc bài hát được sử dụng.

- *music_nused*: Số lần âm thanh đó đã được sử dụng trong các video khác.

- *music_authorName*: The author name of the music. (String)

- *music_originality*: The originality of the music. (String)


## Data Pre-processing

- chuẩn hóa K, M ; xóa đơn vị (videos)
- chuẩn hóa đúng kiểu dữ liệu
- chuẩn hóa vid_duration
- nan processing
- tính viral score 
- tính vid_existtime = vid_scrapetime - vid_scrapetime


## prompting to give a viral score, alt text
## EDA
## model 