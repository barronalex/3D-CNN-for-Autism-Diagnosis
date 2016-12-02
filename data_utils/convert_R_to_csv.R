load("data/ABIDE_fALFF_2.RData")
write.table(fALFF, file = "data/fALFF.csv",row.names=FALSE, na="",col.names=FALSE, sep=",")
write.table(phenotype_data, file = "data/phenotype_data.csv",row.names=FALSE, na="",col.names=FALSE, sep=",")
write.table(coord, file = "data/coord.csv",row.names=FALSE, na="",col.names=FALSE, sep=",")
write.table(region_code, file = "data/region_code.csv",row.names=FALSE, na="",col.names=FALSE, sep=",")
write.table(region_name, file = "data/region_name.csv",row.names=FALSE, na="",col.names=FALSE, sep=",")

load("data/ABIDE_AAL_116_ROI.RData")
for (i in 1:length(AAL_ROI)) {
	# collect names in ones array
	# collect matrices in other array
	id = names(AAL_ROI)[i]
	correlation = AAL_ROI[id]
	correlation = matrix(unlist(correlation))
	path = paste("data/time_series/", id, sep="")
	path = paste(path, ".csv", sep="")
	write.table(correlation, file = path, row.names=FALSE, na="", col.names=FALSE, sep=",")
}
