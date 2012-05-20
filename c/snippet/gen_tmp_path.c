int gen_tmp_path(const char* path, char* tmp_path, const int64_t len)
{
  int err = OB_SUCCESS;
  static volatile uint64_t seq = 0;
  if (NULL == path || NULL == tmp_path || 0 >= len)
  {
    err = OB_INVALID_ARGUMENT;
  }
  else if (0 >= snprintf(tmp_path, len, "%s.%lu.tmp", path, __sync_fetch_and_add(&seq, 1)))
  {
    err = OB_BUF_NOT_ENOUGH;
    TBSYS_LOG(ERROR, "gen_tmp_path failed: path=%s", path);
  }
  return err;
}
