package decoder

import "webtyp.com/fmt"

type decoderError string

func (e decoderError) Error() string {
	return string(e)
}

func makeErr(msg string) error {
	return decoderError(msg)
}

// Error constants for decoder operations.
var (
	ErrInvalidArch             = makeErr("decoder: Config.Arch must be Qwen35 or LFM2")
	ErrLayerKindForArch        = makeErr("decoder: a layer kind does not belong to Config.Arch")
	ErrInvalidVocab            = makeErr("decoder: Config.Vocab must be greater than zero")
	ErrInvalidHidden           = makeErr("decoder: Config.Hidden must be greater than zero")
	ErrInvalidIntermediate     = makeErr("decoder: Config.Intermediate must be greater than zero")
	ErrEmptyLayers             = makeErr("decoder: Config.Layers must not be empty")
	ErrStateMismatch           = makeErr("decoder: the two states come from different models")
	ErrInvalidHeads            = makeErr("decoder: Config.Heads must be greater than zero")
	ErrInvalidKVHeads          = makeErr("decoder: Config.KVHeads must divide Config.Heads")
	ErrInvalidHeadDim          = makeErr("decoder: Config.HeadDim must be greater than zero")
	ErrInvalidRotaryDim        = makeErr("decoder: Config.RotaryDim must be even, positive, and <= Config.HeadDim")
	ErrInvalidRopeTheta        = makeErr("decoder: Config.RopeTheta must be greater than zero")
	ErrInvalidLinearKeyHeads   = makeErr("decoder: Config.LinearKeyHeads must be greater than zero")
	ErrInvalidLinearValueHeads = makeErr("decoder: Config.LinearValueHeads must be a multiple of Config.LinearKeyHeads")
	ErrInvalidLinearKeyDim     = makeErr("decoder: Config.LinearKeyDim must be greater than zero")
	ErrInvalidLinearValueDim   = makeErr("decoder: Config.LinearValueDim must be greater than zero")
	ErrInvalidConvKernel       = makeErr("decoder: Config.ConvKernel must be greater than zero")
	ErrInvalidEps              = makeErr("decoder: Config.Eps must be greater than zero")

	ErrTokenOutOfBounds  = makeErr("decoder: token out of bounds")
	ErrLogitsLenMismatch = makeErr("decoder: logits length mismatch")
	ErrUnsupportedDType  = makeErr("decoder: unsupported tensor dtype")
)

// MissingTensorError returns an error formatted as "decoder: missing tensor <name>".
func MissingTensorError(name string) error {
	return makeErr(fmt.Sprintf("decoder: missing tensor %s", name))
}

// WrongTensorSizeError returns an error formatted as "decoder: tensor <name> has <n> values, want <m>".
func WrongTensorSizeError(name string, got, want int) error {
	return makeErr(fmt.Sprintf("decoder: tensor %s has %s values, want %s", name, fmt.Sprintf("%d", got), fmt.Sprintf("%d", want)))
}

// UnsupportedDTypeError returns an error formatted as "decoder: tensor <name> has dtype <dtype>, which the decoder cannot read".
func UnsupportedDTypeError(name, dtype string) error {
	return fmt.ErrType(makeErr(fmt.Sprintf("decoder: tensor %s has dtype %s, which the decoder cannot read", name, dtype)), ErrUnsupportedDType)
}
